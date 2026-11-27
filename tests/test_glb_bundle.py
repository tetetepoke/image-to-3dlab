"""Download All Animations GLB: one character, every move, one file."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

from image_to_3dlab.glb_bundle import animation_name, bundle, inspect, unique_names
from image_to_3dlab.glb_turn import build_glb, parse_glb

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "viewer"))

import bundle_api


def _character(times: list[float], angles: list[float], *, name: str = "ArmatureAction",
               bones: tuple[str, ...] = ("Hips", "Spine"), vertices: int = 3) -> bytes:
    """A tiny skinned character the way Blender writes one move: mesh, picture, skin, one animation."""
    blob = bytearray()

    def put(data: bytes) -> int:
        blob.extend(b"\0" * (-len(blob) % 4))
        start = len(blob)
        blob.extend(data)
        return start

    views, accessors = [], []

    def accessor(data: bytes, count: int, kind: str, **extra) -> int:
        views.append({"buffer": 0, "byteOffset": put(data), "byteLength": len(data)})
        accessors.append({"bufferView": len(views) - 1, "componentType": 5126, "count": count, "type": kind, **extra})
        return len(accessors) - 1

    position = accessor(struct.pack(f"<{vertices * 3}f", *range(vertices * 3)), vertices, "VEC3",
                        min=[0, 1, 2], max=[6, 7, 8])
    picture = b"\x89PNG fake picture bytes"
    views.append({"buffer": 0, "byteOffset": put(picture), "byteLength": len(picture)})
    image_view = len(views) - 1
    ibm = accessor(struct.pack("<32f", *([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] * 2)), 2, "MAT4")
    t = accessor(struct.pack(f"<{len(times)}f", *times), len(times), "SCALAR", min=[min(times)], max=[max(times)])
    rot = accessor(struct.pack(f"<{len(angles) * 4}f", *[v for a in angles for v in (a, 0, 0, 1)]), len(angles), "VEC4")
    nodes = [{"name": "Armature", "children": [1, 3]}, {"name": bones[0], "children": [2]}, {"name": bones[1]},
             {"name": "Body", "mesh": 0, "skin": 0}]
    doc = {
        "asset": {"version": "2.0", "generator": "test"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": nodes,
        "meshes": [{"primitives": [{"attributes": {"POSITION": position}, "material": 0}]}],
        "materials": [{"name": "Skin", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
        "textures": [{"source": 0}], "images": [{"bufferView": image_view, "mimeType": "image/png"}],
        "skins": [{"joints": [1, 2], "inverseBindMatrices": ibm}],
        "animations": [{"name": name, "samplers": [{"input": t, "output": rot}],
                        "channels": [{"sampler": 0, "target": {"node": 1, "path": "rotation"}},
                                     {"sampler": 0, "target": {"node": 2, "path": "rotation"}}]}],
        "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(blob)}],
    }
    return build_glb(doc, bytes(blob))


def _floats(doc, binary, index):
    acc = doc["accessors"][index]
    view = doc["bufferViews"][acc["bufferView"]]
    n = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[acc["type"]]
    return list(struct.unpack_from(f"<{acc['count'] * n}f", binary, view["byteOffset"] + acc.get("byteOffset", 0)))


IDLE = _character([0, 1, 2], [0.1, 0.2, 0.3])
WALK = _character([0, 0.5], [0.7, 0.8])
JUMP = _character([0, 0.25, 0.5, 0.75], [0.0, 0.4, 0.5, 0.6])


def test_every_move_lands_in_one_file_under_its_own_name():
    data = bundle(IDLE, [("Idle", IDLE), ("Walk", WALK), ("Jump", JUMP)])
    doc, binary = parse_glb(data)
    assert [a["name"] for a in doc["animations"]] == ["Idle", "Walk", "Jump"]
    # shared once: one mesh, one picture, one skin
    assert (len(doc["meshes"]), len(doc["images"]), len(doc["skins"])) == (1, 1, 1)
    walk = doc["animations"][1]
    assert _floats(doc, binary, walk["samplers"][0]["input"]) == [0, 0.5]
    assert _floats(doc, binary, walk["samplers"][0]["output"])[::4] == pytest.approx([0.7, 0.8])
    assert {doc["nodes"][c["target"]["node"]]["name"] for c in walk["channels"]} == {"Hips", "Spine"}
    # the picture survived the rebuild byte for byte
    view = doc["bufferViews"][doc["images"][0]["bufferView"]]
    assert binary[view["byteOffset"]:view["byteOffset"] + view["byteLength"]] == b"\x89PNG fake picture bytes"
    # no copy of the character per move: smaller than three files laid end to end
    assert len(data) < len(IDLE) + len(WALK) + len(JUMP)


def test_the_check_passes_and_reports_what_is_inside():
    report = inspect(bundle(IDLE, [("Idle", IDLE), ("Walk", WALK)]))
    assert report["problems"] == []
    assert report["joints"] == 2
    assert [(a["name"], a["seconds"], a["bones"]) for a in report["animations"]] == [("Idle", 2.0, 2), ("Walk", 0.5, 2)]


def test_the_check_catches_a_generic_name_and_a_missing_bone():
    doc, binary = parse_glb(IDLE)
    doc["animations"][0]["channels"][0]["target"]["node"] = 99
    problems = inspect(build_glb(doc, binary))["problems"]
    assert any("generic name" in p for p in problems)
    assert any("does not exist" in p for p in problems)


def test_a_move_of_another_character_is_refused():
    other = _character([0, 1], [0.1, 0.2], bones=("Root", "Chest"))
    with pytest.raises(ValueError, match="not the same character"):
        bundle(IDLE, [("Idle", IDLE), ("Other", other)])
    with pytest.raises(ValueError, match="share the name"):
        bundle(IDLE, [("Idle", IDLE), ("Idle", WALK)])


def test_names_a_game_can_ask_for():
    labels = {"idle": "Idle (subtle)", "idle_restless": "Idle (restless)", "run_b": "Jog", "hit_react": "Hit react"}
    assert animation_name("sprint") == "Sprint"
    assert unique_names(["idle", "idle_restless", "walk", "run_b", "hit_react"], labels) == [
        "Idle", "IdleRestless", "Walk", "Jog", "HitReact"]


# ------------------------------------------------------------------ the studio's endpoint
def _rig_run(tmp_path: Path) -> Path:
    run = tmp_path / "animate" / "hero__rig__20261127-120000"
    run.mkdir(parents=True)
    (run / "hero_rigged.glb").write_bytes(IDLE)
    for move, data in (("walk", WALK), ("idle", IDLE), ("jump", JUMP)):
        (run / f"hero_{move}.glb").write_bytes(data)
    (run / "hero_rigged_try1.glb").write_bytes(IDLE)     # a kept first try is not a move
    return run


def test_the_endpoint_bundles_every_move_calm_ones_first(tmp_path):
    run = _rig_run(tmp_path)
    made = bundle_api.make_bundle("animate/hero__rig__20261127-120000/hero_rigged.glb", tmp_path)
    assert made["file"] == "animate/hero__rig__20261127-120000/bundle/hero_all_moves.glb"
    assert [a["name"] for a in made["animations"]] == ["Idle", "Walk", "Jump"]
    assert made["count"] == 3 and made["bytes"] == (run / "bundle" / "hero_all_moves.glb").stat().st_size
    assert made["check"]["problems"] == []
    assert (run / "bundle" / "hero_all_moves.provenance.json").is_file()


def test_the_endpoint_reuses_the_file_until_a_move_is_added(tmp_path):
    run = _rig_run(tmp_path)
    rel = "animate/hero__rig__20261127-120000/hero_rigged.glb"
    bundle_api.make_bundle(rel, tmp_path)
    out = run / "bundle" / "hero_all_moves.glb"
    first = out.stat().st_mtime_ns
    assert bundle_api.make_bundle(rel, tmp_path)["count"] == 3
    assert out.stat().st_mtime_ns == first
    (run / "hero_kick.glb").write_bytes(_character([0, 1], [0.9, 1.0]))
    assert [a["name"] for a in bundle_api.make_bundle(rel, tmp_path)["animations"]] == ["Idle", "Walk", "Jump", "Kick"]


def test_the_endpoint_only_takes_rigged_models(tmp_path):
    _rig_run(tmp_path)
    for bad in ("../secret_rigged.glb", "animate/hero__rig__20261127-120000/hero_walk.glb", "nope_rigged.glb"):
        with pytest.raises(ValueError):
            bundle_api.make_bundle(bad, tmp_path)
