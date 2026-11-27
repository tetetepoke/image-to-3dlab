"""Put every move of one character into a single GLB.

Each Animate run writes one GLB per move, each a full copy of the character with one
animation, usually named `ArmatureAction`. Games want one file per character: one mesh,
one set of textures and materials, one skeleton and its skin weights, and every move as
its own named animation. Three.js plays them with `AnimationMixer`; Blender imports each
as its own action.

Pure Python, no Blender: the moves of one rig share their node names, so an animation is
moved across by pointing each channel at the node of the same name in the base file.
"""

from __future__ import annotations

import re
import struct
from itertools import pairwise
from typing import Any

from image_to_3dlab.glb_turn import build_glb, parse_glb

PATH_TYPES = {"translation": "VEC3", "rotation": "VEC4", "scale": "VEC3", "weights": "SCALAR"}
COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
BYTES = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}


def animation_name(clip_id: str, labels: dict[str, str] | None = None) -> str:
    """The name a game asks for: `Idle`, `Walk`, `Sprint`, `HitReact`.

    The move's label wins over its file id ("run_b" is labelled "Jog"). A bracketed detail
    is dropped ("Idle (subtle)" -> "Idle"); `unique_names` brings it back on a clash.
    """
    label = (labels or {}).get(clip_id) or clip_id
    label = re.sub(r"\(.*?\)", "", label)
    words = re.findall(r"[A-Za-z0-9]+", label.replace("_", " "))
    return "".join(w[:1].upper() + w[1:] for w in words) or "Move"


def unique_names(clip_ids: list[str], labels: dict[str, str] | None = None) -> list[str]:
    """One name per move, no two alike. A clash keeps its bracketed detail, then a number."""
    names: list[str] = []
    for clip_id in clip_ids:
        name = animation_name(clip_id, labels)
        if name in names:
            full = (labels or {}).get(clip_id) or clip_id
            detail = "".join(w[:1].upper() + w[1:] for w in re.findall(r"[A-Za-z0-9]+", " ".join(re.findall(r"\((.*?)\)", full))))
            name = name + detail if detail and name + detail not in names else name
        base, n = name, 2
        while name in names:
            name, n = f"{base}{n}", n + 1
        names.append(name)
    return names


# ------------------------------------------------------------------ buffer bookkeeping
class _Writer:
    """Builds the new BIN chunk view by view, 4-byte aligned."""

    def __init__(self) -> None:
        self.data = bytearray()
        self.views: list[dict[str, Any]] = []

    def add_view(self, view: dict[str, Any], binary: bytes) -> int:
        if "buffer" in view and view["buffer"] != 0:
            raise ValueError("only single-buffer GLB files can be bundled")
        if view.get("extensions"):
            raise ValueError("compressed GLB files (meshopt) cannot be bundled; bundle before compressing")
        start = view.get("byteOffset", 0)
        chunk = binary[start:start + view["byteLength"]]
        self.data += b"\0" * (-len(self.data) % 4)
        new = {k: v for k, v in view.items() if k not in ("buffer", "byteOffset")}
        new.update(buffer=0, byteOffset=len(self.data))
        self.data += chunk
        self.views.append(new)
        return len(self.views) - 1


def _mesh_and_skin_accessors(doc: dict[str, Any]) -> set[int]:
    used: set[int] = set()
    for mesh in doc.get("meshes", []):
        for prim in mesh.get("primitives", []):
            used.update(prim.get("attributes", {}).values())
            if "indices" in prim:
                used.add(prim["indices"])
            for target in prim.get("targets", []):
                used.update(target.values())
    for skin in doc.get("skins", []):
        if "inverseBindMatrices" in skin:
            used.add(skin["inverseBindMatrices"])
    return used


def _strip_animations(doc: dict[str, Any], binary: bytes) -> tuple[dict[str, Any], _Writer]:
    """The base file with its animations removed and its BIN chunk rebuilt without them."""
    doc = {k: v for k, v in doc.items() if k != "animations"}
    keep_acc = sorted(_mesh_and_skin_accessors(doc))
    writer = _Writer()
    view_map: dict[int, int] = {}

    def view(i: int) -> int:
        if i not in view_map:
            view_map[i] = writer.add_view(doc["bufferViews"][i], binary)
        return view_map[i]

    acc_map: dict[int, int] = {}
    accessors = []
    for i in keep_acc:
        acc = dict(doc["accessors"][i])
        if "bufferView" in acc:
            acc["bufferView"] = view(acc["bufferView"])
        if "sparse" in acc:
            sp = acc["sparse"] = {**acc["sparse"]}
            sp["indices"] = {**sp["indices"], "bufferView": view(sp["indices"]["bufferView"])}
            sp["values"] = {**sp["values"], "bufferView": view(sp["values"]["bufferView"])}
        acc_map[i] = len(accessors)
        accessors.append(acc)
    for image in doc.get("images", []):
        if "bufferView" in image:
            image["bufferView"] = view(image["bufferView"])
    for mesh in doc.get("meshes", []):
        for prim in mesh.get("primitives", []):
            prim["attributes"] = {k: acc_map[v] for k, v in prim.get("attributes", {}).items()}
            if "indices" in prim:
                prim["indices"] = acc_map[prim["indices"]]
            if "targets" in prim:
                prim["targets"] = [{k: acc_map[v] for k, v in t.items()} for t in prim["targets"]]
    for skin in doc.get("skins", []):
        if "inverseBindMatrices" in skin:
            skin["inverseBindMatrices"] = acc_map[skin["inverseBindMatrices"]]
    doc["accessors"] = accessors
    return doc, writer


def _node_names(doc: dict[str, Any]) -> list[str | None]:
    return [n.get("name") for n in doc.get("nodes", [])]


def _shape_signature(doc: dict[str, Any]) -> tuple:
    """What must match for two files to be the same character: nodes, skin joints, mesh sizes."""
    names = _node_names(doc)
    joints = tuple(tuple(names[j] for j in s.get("joints", [])) for s in doc.get("skins", []))
    counts = tuple(doc["accessors"][p["attributes"]["POSITION"]].get("count")
                   for m in doc.get("meshes", []) for p in m.get("primitives", []) if "POSITION" in p.get("attributes", {}))
    return tuple(names), joints, counts


class _AccessorCopier:
    """Copies one clip's keyframe accessors (and their data) into the new file, each once."""

    def __init__(self, doc, writer, cdoc, cbin, name, acc_map, view_map) -> None:
        self.doc, self.writer, self.cdoc, self.cbin, self.name = doc, writer, cdoc, cbin, name
        self.acc_map, self.view_map = acc_map, view_map

    def __call__(self, i: int) -> int:
        if i not in self.acc_map:
            acc = dict(self.cdoc["accessors"][i])
            if "sparse" in acc:
                raise ValueError(f"move {self.name!r} uses sparse keyframes, which cannot be bundled")
            v = acc["bufferView"]
            if v not in self.view_map:
                self.view_map[v] = self.writer.add_view(self.cdoc["bufferViews"][v], self.cbin)
            acc["bufferView"] = self.view_map[v]
            self.acc_map[i] = len(self.doc["accessors"])
            self.doc["accessors"].append(acc)
        return self.acc_map[i]


# ------------------------------------------------------------------ bundle
def bundle(base: bytes, clips: list[tuple[str, bytes]]) -> bytes:
    """One GLB: the character of `base` with the animations of every clip, named as given.

    `clips` is a list of (animation name, GLB bytes). Every clip must be the same character
    (same nodes, skin joints and mesh sizes) as `base`, or ValueError says which one is not.
    """
    if not clips:
        raise ValueError("no moves to bundle")
    base_doc, base_bin = parse_glb(base)
    signature = _shape_signature(base_doc)
    doc, writer = _strip_animations(base_doc, base_bin)
    by_name: dict[str, int] = {}
    for i, name in enumerate(_node_names(doc)):
        if name is not None:
            if name in by_name:
                raise ValueError(f"two nodes share the name {name!r}; moves cannot be matched to them")
            by_name[name] = i
    animations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name, data in clips:
        if name in seen:
            raise ValueError(f"two moves share the name {name!r}")
        seen.add(name)
        cdoc, cbin = parse_glb(data)
        if _shape_signature(cdoc) != signature:
            raise ValueError(f"move {name!r} is not the same character (its skeleton or mesh differs)")
        canims = cdoc.get("animations", [])
        if not canims:
            raise ValueError(f"move {name!r} has no animation in it")
        cnames = _node_names(cdoc)
        for k, anim in enumerate(canims):
            view_map: dict[int, int] = {}
            acc_map: dict[int, int] = {}
            copy = _AccessorCopier(doc, writer, cdoc, cbin, name, acc_map, view_map)
            samplers = [{**s, "input": copy(s["input"]), "output": copy(s["output"])} for s in anim["samplers"]]
            channels = []
            for ch in anim["channels"]:
                target = dict(ch["target"])
                node_name = cnames[target["node"]]
                if node_name not in by_name:
                    raise ValueError(f"move {name!r} animates {node_name!r}, which the character does not have")
                target["node"] = by_name[node_name]
                channels.append({**ch, "target": target})
            out = {"name": name if len(canims) == 1 else f"{name}{k + 1}", "samplers": samplers, "channels": channels}
            if anim.get("extras"):
                out["extras"] = anim["extras"]
            animations.append(out)
    doc["animations"] = animations
    doc["bufferViews"] = writer.views
    doc["buffers"] = [{"byteLength": len(writer.data) + (-len(writer.data) % 4)}]
    return build_glb(doc, bytes(writer.data))


# ------------------------------------------------------------------ check
def _read_floats(doc: dict[str, Any], binary: bytes, i: int) -> list[float]:
    acc = doc["accessors"][i]
    view = doc["bufferViews"][acc["bufferView"]]
    n = COMPONENTS[acc["type"]]
    stride = view.get("byteStride") or n * 4
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    out = []
    for e in range(acc["count"]):
        out.extend(struct.unpack_from(f"<{n}f", binary, start + e * stride))
    return out


def inspect(data: bytes) -> dict[str, Any]:
    """What is inside a bundled GLB, and whether every animation points at real bones.

    Returns counts (meshes, materials, textures, skins, joints), one row per animation
    (name, channels, seconds, bones it moves) and `problems`, empty when all is well.
    """
    doc, binary = parse_glb(data)
    problems: list[str] = []
    nodes = doc.get("nodes", [])
    accessors = doc.get("accessors", [])
    views = doc.get("bufferViews", [])
    joints = {j for s in doc.get("skins", []) for j in s.get("joints", [])}
    for k, skin in enumerate(doc.get("skins", [])):
        for j in skin.get("joints", []):
            if not 0 <= j < len(nodes):
                problems.append(f"skin {k} lists bone {j}, which does not exist")
        ibm = skin.get("inverseBindMatrices")
        if ibm is not None and accessors[ibm]["count"] != len(skin.get("joints", [])):
            problems.append(f"skin {k}: {accessors[ibm]['count']} bind matrices for {len(skin['joints'])} bones")
    for v in views:
        if v.get("byteOffset", 0) + v["byteLength"] > len(binary):
            problems.append("a buffer view runs past the end of the file")
            break
    rows = []
    names_seen: set[str] = set()
    for a, anim in enumerate(doc.get("animations", [])):
        name = anim.get("name") or f"#{a}"
        if name in names_seen:
            problems.append(f"two animations are both called {name!r}")
        names_seen.add(name)
        if re.fullmatch(r"(Armature)?Action(\.\d+)?", name):
            problems.append(f"animation {a} still has the generic name {name!r}")
        seconds, moved = 0.0, set()
        samplers = anim.get("samplers", [])
        for c, ch in enumerate(anim.get("channels", [])):
            t = ch.get("target", {})
            node, path = t.get("node"), t.get("path")
            if node is None or not 0 <= node < len(nodes):
                problems.append(f"{name}: channel {c} points at a bone that does not exist")
                continue
            if path not in PATH_TYPES:
                problems.append(f"{name}: channel {c} has unknown path {path!r}")
                continue
            # moving a node that is not a bone (the armature itself) is fine if bones hang under it
            if path != "weights" and node not in joints and not _is_ancestor_of_joint(nodes, node, joints):
                problems.append(f"{name}: channel {c} moves {nodes[node].get('name')!r}, which is not part of the skeleton")
            s = ch.get("sampler")
            if s is None or not 0 <= s < len(samplers):
                problems.append(f"{name}: channel {c} has no sampler")
                continue
            sm = samplers[s]
            inp, out = accessors[sm["input"]], accessors[sm["output"]]
            per = 3 if sm.get("interpolation") == "CUBICSPLINE" else 1
            if path != "weights":
                if out["type"] != PATH_TYPES[path]:
                    problems.append(f"{name}: {path} keys of {nodes[node].get('name')!r} have type {out['type']}")
                if out["count"] != inp["count"] * per:
                    problems.append(f"{name}: {nodes[node].get('name')!r} has {inp['count']} times but {out['count']} keys")
            times = _read_floats(doc, binary, sm["input"])
            if any(b < a2 for a2, b in pairwise(times)):
                problems.append(f"{name}: key times of {nodes[node].get('name')!r} go backwards")
            seconds = max(seconds, max(times) if times else 0.0)
            moved.add(node)
        rows.append({"name": name, "channels": len(anim.get("channels", [])), "seconds": round(seconds, 3),
                     "bones": len(moved & joints)})
    return {
        "meshes": len(doc.get("meshes", [])),
        "materials": len(doc.get("materials", [])),
        "textures": len(doc.get("textures", [])),
        "images": len(doc.get("images", [])),
        "skins": len(doc.get("skins", [])),
        "joints": len(joints),
        "animations": rows,
        "problems": problems,
        "bytes": len(data),
    }


def _is_ancestor_of_joint(nodes: list[dict[str, Any]], node: int, joints: set[int]) -> bool:
    stack = list(nodes[node].get("children", []))
    while stack:
        n = stack.pop()
        if n in joints:
            return True
        stack.extend(nodes[n].get("children", []))
    return False
