"""Write calibration results back into a URDF: fold a perturbation dict into the joint origins.

The perturbation keys are the ones `urdf_kinematics.UrdfTree` understands (see its module docstring). The patched
URDF gives, for the same measured joint positions `q`, exactly the poses `UrdfTree(original).link_pose(link, q,
perturb)` gives, so it can be used by anything that reads a URDF (placo, a viewer, the numpy FK) without knowing
about the calibration:

- `pos:<joint>:x|y|z` is added to the joint's origin translation.
- `rot:<joint>:roll|pitch|yaw` becomes an extra rotation after the origin rotation, folded into its rpy.
- `zero:<joint>` (a joint zero offset) becomes a fixed rotation about the joint's axis (translation along it for a
  prismatic joint), also folded into the origin. The joint limits are left as they are: they are limits on the
  measured joint position, which does not change.

Only joint origins are rewritten, so the result keeps everything else in the file (meshes, inertias, ...).
"""

import xml.etree.ElementTree as ET

import numpy as np

from lerobot.felix_jr.calibration.urdf_kinematics import (
    rotation_about_axis,
    rotation_from_rpy,
    rpy_from_rotation,
)


def patch_urdf(xml: str, perturb: dict[str, float]) -> str:
    """The URDF text `xml` with `perturb` folded into the joint origins. Raises KeyError for an unknown joint."""
    root = ET.fromstring(xml)
    joints = {j.get("name"): j for j in root.findall("joint")}
    by_joint: dict[str, dict[str, float]] = {}
    for key, value in perturb.items():
        kind, joint, *rest = key.split(":")
        if joint not in joints:
            raise KeyError(f"{key!r}: no joint {joint!r} in the URDF")
        by_joint.setdefault(joint, {})[key] = value
    for name, values in by_joint.items():
        element = joints[name]
        origin = element.find("origin")
        if origin is None:
            origin = ET.SubElement(element, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        xyz = _floats(origin.get("xyz", "0 0 0"))
        rpy = _floats(origin.get("rpy", "0 0 0"))
        axis_el = element.find("axis")
        axis = _floats(axis_el.get("xyz")) if axis_el is not None else np.array([1.0, 0.0, 0.0])
        axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 0 else axis  # fixed joints may have none

        xyz = xyz + np.array([values.get(f"pos:{name}:{a}", 0.0) for a in ("x", "y", "z")])
        rotation = rotation_from_rpy(*rpy)
        extra = [values.get(f"rot:{name}:{a}", 0.0) for a in ("roll", "pitch", "yaw")]
        if any(extra):
            rotation = rotation @ rotation_from_rpy(*extra)
        zero = values.get(f"zero:{name}", 0.0)
        if zero:
            kind = element.get("type")
            if kind in ("revolute", "continuous"):
                rotation = rotation @ rotation_about_axis(axis, zero)
            elif kind == "prismatic":
                xyz = xyz + rotation @ (axis * zero)
        origin.set("xyz", _format(xyz))
        origin.set("rpy", _format(np.array(rpy_from_rotation(rotation))))
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def _floats(text: str) -> np.ndarray:
    return np.array([float(v) for v in text.split()], dtype=float)


def _format(values: np.ndarray) -> str:
    return " ".join(f"{v:.12g}" for v in values)
