"""Build `xlerobot_so101.urdf`: the XLeRobot's head and base with two SO-101 arms, as one kinematic tree.

`XleRobotModel` (`xle_robot_model.py`) loads this one URDF. The arms come from the official SO-101 URDF rather than
from `xlerobot.urdf`, whose arm is an older gripper (its wrist roll joint lacks the SO-101's 18 mm lateral offset,
and its shoulder offsets differ by about 13 mm), while the hardware is an SO-101; and the SO-101 URDF's joint angles
are LeRobot's calibrated degrees one to one (the teleop relies on this), so no sign or offset is guessed for the arm:

- From `xlerobot.urdf`: everything except its arms. The root `joint_0` becomes fixed (pinocchio and placo would
  add a free base otherwise), and the mesh-only `*_geom_*` links and the `head_camera_rgb_frame` /
  `head_camera_depth_frame` links (their rotation puts the optical axis sideways) are dropped.
- Two copies of `so101_new_calib.urdf`, every link and joint prefixed `left_` / `right_`, each attached to its
  arm base (`Base` / `Base_2`) by a fixed `<arm>_arm_mount` joint. Its origin is the rigid transform that best
  matches the SO-101's shoulder, elbow and wrist pivots with those of the arm inside `xlerobot.urdf`, over random
  poses (the alignment the first calibrations were made with, kept so they still apply).
- A fixed `head_camera_optical` frame under `head_camera_link`: x right, y down, z forward.
- No visuals, collisions, transmissions or materials: we do not have the meshes, and only the tree is needed.

The output is committed; a test checks it is what this script builds. Rebuild it with
`uv run python -m lerobot.felix_jr.calibration.merge_urdf`.
"""

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from lerobot.utils.import_utils import _pinocchio_available, require_package

if TYPE_CHECKING or _pinocchio_available:
    import pinocchio as pin

FELIX_DIR = Path(__file__).resolve().parent.parent
XLEROBOT_URDF = FELIX_DIR / "xlerobot.urdf"
SO101_URDF = FELIX_DIR / "so101_new_calib.urdf"
MERGED_URDF = FELIX_DIR / "xlerobot_so101.urdf"

ARMS = ("left", "right")
ARM_BASES = {"left": "Base", "right": "Base_2"}
CAMERA_LINK, OPTICAL_FRAME = "head_camera_link", "head_camera_optical"
# Optical frame axes in the camera link's frame: x right (-y), y down (-z), z forward (+x).
LINK_TO_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
DROPPED_LINKS = ("head_camera_rgb_frame", "head_camera_depth_frame")

# How the SO-101's joints relate to the arm joints inside xlerobot.urdf: q_xle = sign * q_so101 + offset. Signs and
# the lift / elbow / wrist-flex offsets were found by matching the pivots' positions over random poses (the residual
# that remains, about 8 to 14 mm, is the shoulder geometry difference). The pan offset is the heading: LeRobot's
# pan = 0 points the arm forward, while the XLeRobot URDF's pan = 0 points it sideways (left arm 90 degrees off,
# right arm -90). Checked against the wiggle test's camera frames: a +10 degree pan moves the arms to the right in
# the image, and at the center pose the tip is about 35 cm forward and 23 cm up from the arm base.
SO101_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
XLE_ARM_JOINTS = {
    "left": ("Rotation_L", "Pitch_L", "Elbow_L", "Wrist_Pitch_L"),
    "right": ("Rotation_R", "Pitch_R", "Elbow_R", "Wrist_Pitch_R"),
}
ALIGN_SIGNS = (1.0, -1.0, 1.0, 1.0)
ALIGN_OFFSETS = {
    "left": (np.pi / 2, np.pi / 2, np.pi / 2, 0.0),
    "right": (-np.pi / 2, np.pi / 2, np.pi / 2, 0.0),
}


def build_merged_urdf() -> str:
    """The merged URDF as text (see the module docstring)."""
    require_package("pin", extra="kinematics", import_name="pinocchio")
    xle = ET.fromstring(XLEROBOT_URDF.read_text())
    _joint(xle, "joint_0").set("type", "fixed")
    mounts = {arm: arm_mount(xle, arm) for arm in ARMS}

    old_arm_links = set().union(*(_descendants(xle, ARM_BASES[arm]) for arm in ARMS))
    geom_links = {
        attr(link, "name") for link in xle.findall("link") if re.search(r"_geom_\d+$", attr(link, "name"))
    }
    _remove_links(xle, old_arm_links | geom_links | set(DROPPED_LINKS))
    for link in xle.findall("link"):
        _strip_geometry(link)

    so101 = ET.fromstring(SO101_URDF.read_text())
    for arm in ARMS:
        prefix = f"{arm}_"
        for link in so101.findall("link"):
            copy = ET.fromstring(ET.tostring(link))
            copy.set("name", prefix + attr(link, "name"))
            _strip_geometry(copy)
            xle.append(copy)
        for joint in so101.findall("joint"):
            copy = ET.fromstring(ET.tostring(joint))
            copy.set("name", prefix + attr(joint, "name"))
            for end in ("parent", "child"):
                child_element(copy, end).set("link", prefix + link_of(joint, end))
            xle.append(copy)
        _add_fixed_joint(xle, f"{arm}_arm_mount", ARM_BASES[arm], f"{prefix}base_link", mounts[arm])

    xle.append(ET.Element("link", {"name": OPTICAL_FRAME}))
    _add_fixed_joint(
        xle, f"{OPTICAL_FRAME}_joint", CAMERA_LINK, OPTICAL_FRAME, pin.SE3(LINK_TO_OPTICAL, np.zeros(3))
    )

    xle.set("name", "xlerobot_so101")
    xle.insert(
        0,
        ET.Comment(
            " Generated by lerobot.felix_jr.calibration.merge_urdf from xlerobot.urdf and so101_new_calib.urdf"
            " (Apache License 2.0, TheRobotStudio/SO-ARM100). Do not edit; rebuild it. "
        ),
    )
    ET.indent(xle)
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(xle, encoding="unicode") + "\n"


def arm_mount(xle: ET.Element, arm: str) -> "pin.SE3":
    """X_Base_so101: where the SO-101's `base_link` sits in the XLeRobot arm's base frame (`Base` / `Base_2`).

    Kabsch fit of the SO-101's shoulder, elbow and wrist pivots onto the same pivots of the arm in
    `xlerobot.urdf`, over 16 random poses. Changing the poses or the fit changes the arm's mount, and with it the
    meaning of saved calibrations: a test checks the URDF still reproduces the first calibrations' model.
    """
    xle_model = _model(ET.tostring(xle, encoding="unicode"))
    so_model = _model(SO101_URDF.read_text())
    so_pivots = [_child(so_model, j) for j in SO101_JOINTS[1:4]]
    xle_joints = XLE_ARM_JOINTS[arm]
    xle_pivots = [_child(xle_model, j) for j in xle_joints[1:4]]
    x_base_root = _link_pose(xle_model, ARM_BASES[arm], {}).inverse()
    rng = np.random.default_rng(0)
    src, dst = [], []
    for _ in range(16):
        q_so = {j: rng.uniform(-1.0, 1.0) for j in SO101_JOINTS}
        q_xle = {
            xj: sign * q_so[sj] + offset
            for xj, sj, sign, offset in zip(
                xle_joints, SO101_JOINTS[:4], ALIGN_SIGNS, ALIGN_OFFSETS[arm], strict=True
            )
        }
        for so_link, xle_link in zip(so_pivots, xle_pivots, strict=True):
            src.append(_link_pose(so_model, so_link, q_so).translation)
            dst.append((x_base_root * _link_pose(xle_model, xle_link, q_xle)).translation)
    src_points, dst_points = np.array(src), np.array(dst)
    centroid_src, centroid_dst = src_points.mean(0), dst_points.mean(0)
    u, _, vt = np.linalg.svd((src_points - centroid_src).T @ (dst_points - centroid_dst))
    rotation = vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    return pin.SE3(rotation, centroid_dst - rotation @ centroid_src)


def _model(xml: str) -> "pin.Model":
    return pin.buildModelFromXML(xml.replace('type="floating"', 'type="fixed"'))


def _child(model: "pin.Model", joint: str) -> str:
    """The child link of a moving joint (the body frame whose parent is that joint)."""
    joint_id = model.getJointId(joint)
    return next(f.name for f in model.frames if f.type == pin.FrameType.BODY and f.parentJoint == joint_id)


def _link_pose(model: "pin.Model", link: str, q: dict[str, float]) -> "pin.SE3":
    config = pin.neutral(model)
    for name, value in q.items():
        config[model.joints[model.getJointId(name)].idx_q] = value
    data = model.createData()
    pin.framesForwardKinematics(model, data, config)
    return data.oMf[model.getFrameId(link, pin.FrameType.BODY)]


def _joint(root: ET.Element, name: str) -> ET.Element:
    return next(j for j in root.findall("joint") if j.get("name") == name)


def _descendants(root: ET.Element, link: str) -> set[str]:
    """Links below `link` (not including it)."""
    children: dict[str, list[str]] = {}
    for joint in root.findall("joint"):
        children.setdefault(link_of(joint, "parent"), []).append(link_of(joint, "child"))
    out: set[str] = set()
    stack = list(children.get(link, []))
    while stack:
        name = stack.pop()
        out.add(name)
        stack.extend(children.get(name, []))
    return out


def _remove_links(root: ET.Element, links: set[str]) -> None:
    """Remove `links` and the joints into them."""
    for joint in root.findall("joint"):
        if link_of(joint, "child") in links:
            root.remove(joint)
    for link in root.findall("link"):
        if link.get("name") in links:
            root.remove(link)


def attr(element: ET.Element, key: str) -> str:
    """An attribute that must be there."""
    value = element.get(key)
    if value is None:
        raise ValueError(f"<{element.tag}> has no {key!r} attribute")
    return value


def child_element(element: ET.Element, tag: str) -> ET.Element:
    """A sub-element that must be there."""
    found = element.find(tag)
    if found is None:
        raise ValueError(f"<{element.tag} name={element.get('name')!r}> has no <{tag}>")
    return found


def link_of(joint: ET.Element, end: str) -> str:
    """The link at a joint's `end` ("parent" or "child")."""
    return attr(child_element(joint, end), "link")


def _strip_geometry(link: ET.Element) -> None:
    for child in list(link):
        if child.tag != "inertial":
            link.remove(child)


def _add_fixed_joint(root: ET.Element, name: str, parent: str, child: str, origin: "pin.SE3") -> None:
    joint = ET.SubElement(root, "joint", {"name": name, "type": "fixed"})
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})
    rpy = pin.rpy.matrixToRpy(origin.rotation)
    ET.SubElement(joint, "origin", {"xyz": _format(origin.translation), "rpy": _format(rpy)})


def _format(values: np.ndarray) -> str:
    return " ".join(f"{v:.15g}" for v in values)


def main() -> None:
    """Command line entry point: write the merged URDF (see the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=MERGED_URDF)
    args = parser.parse_args()
    args.out.write_text(build_merged_urdf())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
