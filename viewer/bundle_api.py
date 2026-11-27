"""Download All Animations GLB: every move of one character, in one file.

Animate writes one GLB per move beside the rigged model. This puts them together:

    output/animate/<name>__rig__<stamp>/
        <name>_rigged.glb
        <name>_idle.glb, <name>_walk.glb, ...       one per move (unchanged)
        bundle/<name>_all_moves.glb                 one mesh, one skeleton, every move by name
        bundle/<name>_all_moves.provenance.json     the licences of every move in it

The bundle sits in its own folder so the Library never mistakes it for another move. It is
made again only when a move was added or changed since.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from assets_api import _is_second_take

from image_to_3dlab.glb_bundle import bundle, inspect, unique_names
from image_to_3dlab.provenance import set_glb_generator, write_provenance

OUTPUT_ROOT = REPO / "output"
PRESETS = REPO / "image_to_3dlab" / "motions" / "presets.json"
BUNDLE_DIR = "bundle"
SUFFIX = "_all_moves"
# Calm moves first, so a game that just plays "the first animation" gets something sensible.
FIRST = ("idle", "idle_restless", "walk")


def _presets(index: Path = PRESETS) -> list[dict[str, Any]]:
    try:
        return json.loads(index.read_text(encoding="utf-8")).get("presets") or []
    except (OSError, ValueError):
        return []


def rig_run(rigged_relative: str, output: Path = OUTPUT_ROOT) -> Path:
    """The rigged model the page asked about, if it really is one inside output/animate."""
    rigged = (output / rigged_relative).resolve()
    animate = (output / "animate").resolve()
    if animate not in rigged.parents or not rigged.name.endswith("_rigged.glb") or not rigged.is_file():
        raise ValueError(f"not a rigged model: {rigged_relative!r}")
    return rigged


def moves_of(rigged: Path, index: Path = PRESETS) -> list[tuple[str, Path]]:
    """(move id, file) for every move made on this rig, in the order the studio lists moves."""
    stem = rigged.name.removesuffix("_rigged.glb")
    found = {}
    for clip in rigged.parent.glob(f"{stem}_*.glb"):
        move = clip.stem.removeprefix(f"{stem}_")
        # the same moves the Library lists: not the rig itself, not a kept first try, not a second take
        if clip == rigged or move.startswith("rigged") or _is_second_take(move):
            continue
        found[move] = clip
    order = [p["id"] for p in _presets(index)]
    rank = {m: (FIRST.index(m) if m in FIRST else len(FIRST) + (order.index(m) if m in order else len(order)), m)
            for m in found}
    return [(m, found[m]) for m in sorted(found, key=lambda m: rank[m])]


def bundle_path(rigged: Path) -> Path:
    return rigged.parent / BUNDLE_DIR / f"{rigged.name.removesuffix('_rigged.glb')}{SUFFIX}.glb"


def _fresh(out: Path, sources: list[Path]) -> bool:
    if not out.is_file():
        return False
    record = out.with_suffix(".provenance.json")
    try:
        listed = json.loads(record.read_text(encoding="utf-8"))["moves"]
    except (OSError, ValueError, KeyError):
        return False
    have = {m["file"]: m["sha256"] for m in listed}
    return len(have) == len(sources) and all(
        have.get(s.name) == hashlib.sha256(s.read_bytes()).hexdigest() for s in sources)


def make_bundle(rigged_relative: str, output: Path = OUTPUT_ROOT, index: Path = PRESETS) -> dict[str, Any]:
    """Build (or reuse) the all-moves GLB of one rigged model and say what is in it."""
    rigged = rig_run(rigged_relative, output)
    moves = moves_of(rigged, index)
    if not moves:
        raise ValueError("this character has no moves yet; add one with Animate first")
    labels = {p["id"]: p.get("label", p["id"]) for p in _presets(index)}
    names = unique_names([m for m, _ in moves], labels)
    out = bundle_path(rigged)
    if not _fresh(out, [f for _, f in moves]):
        datas = [f.read_bytes() for _, f in moves]
        data = bundle(datas[0], list(zip(names, datas, strict=True)))
        report = inspect(data)
        if report["problems"]:
            raise ValueError("the bundled file failed its check: " + "; ".join(report["problems"][:3]))
        out.parent.mkdir(exist_ok=True)
        temporary = out.with_name(f".{out.name}.tmp")
        temporary.write_bytes(data)
        set_glb_generator(temporary)
        temporary.replace(out)
        record = _read(rigged.with_suffix(".provenance.json")) or {}
        write_provenance(out.with_suffix(".provenance.json"), {
            "schema_version": 1,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "tool": "viewer: Download All Animations GLB",
            "license": record.get("license"),
            "classification": record.get("classification"),
            "rig": record.get("rig"),
            "source_record": record.get("source_record"),
            "moves": [{"name": n, "move": m, "file": f.name, "sha256": hashlib.sha256(d).hexdigest(),
                       "motion": (_read(f.with_suffix(".provenance.json")) or {}).get("motion")}
                      for n, (m, f), d in zip(names, moves, datas, strict=True)],
        }, atomic=True)
    data = out.read_bytes()
    report = inspect(data)
    return {
        "file": out.relative_to(output).as_posix(),
        "count": len(report["animations"]),
        "bytes": len(data),
        "animations": report["animations"],
        "moves": [{"name": n, "move": m} for n, (m, _) in zip(names, moves, strict=True)],
        "check": {k: report[k] for k in ("meshes", "materials", "textures", "images", "skins", "joints", "problems")},
    }


def _read(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None
