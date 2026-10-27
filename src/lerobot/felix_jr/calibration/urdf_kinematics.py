"""Forward kinematics straight from a URDF, in numpy, with small parameter perturbations for calibration work.

Only the kinematic tree is read (joint origins, axes, types, limits); meshes are not needed. Poses are 4x4
homogeneous matrices `X_AB` (frame B in frame A, Drake notation: `X_AC = X_AB @ X_BC`); everything is relative
to the URDF's root link.

Perturbations are a dict `{parameter name: value}` applied on top of the URDF values:

- `zero:<joint>` (rad): added to that joint's angle (a joint zero offset).
- `pos:<joint>:x|y|z` (m): added to the joint's origin translation, in the parent link's frame.
- `rot:<joint>:roll|pitch|yaw` (rad): an extra rotation of the joint's origin about the joint frame's own
  x / y / z axes, applied after the URDF's origin rotation (e.g. a camera mount that is slightly tilted).
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

AXIS_INDEX = {"x": 0, "y": 1, "z": 2, "roll": 0, "pitch": 1, "yaw": 2}


def rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr, cp, sp, cy, sy = (
        np.cos(roll),
        np.sin(roll),
        np.cos(pitch),
        np.sin(pitch),
        np.cos(yaw),
        np.sin(yaw),
    )
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues: rotation by `angle` (rad) about unit `axis`."""
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx


def pose(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """4x4 homogeneous transform from a 3x3 rotation and a translation."""
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = translation
    return out


def invert(x_ab: np.ndarray) -> np.ndarray:
    """X_BA from X_AB."""
    r, t = x_ab[:3, :3], x_ab[:3, 3]
    return pose(r.T, -r.T @ t)


@dataclass(frozen=True)
class Joint:
    """One URDF joint: its type, the links it connects, its origin (xyz, rpy), axis and limits."""

    name: str
    type: str  # revolute, continuous, prismatic, fixed, floating
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray
    lower: float | None = None
    upper: float | None = None

    @property
    def moves(self) -> bool:
        """True for joints with a position (revolute, continuous, prismatic)."""
        return self.type in ("revolute", "continuous", "prismatic")


class UrdfTree:
    """The kinematic tree of a URDF. Floating joints (e.g. a mobile base's root) are treated as fixed."""

    def __init__(self, path: str | Path | None = None, xml: str | None = None):
        """Read the tree from a URDF file (`path`) or from a URDF string (`xml`)."""
        if (path is None) == (xml is None):
            raise ValueError("pass exactly one of `path` or `xml`")
        root = ET.parse(path).getroot() if path is not None else ET.fromstring(xml)
        self.joints: dict[str, Joint] = {}
        for j in root.findall("joint"):
            origin, axis, limit = j.find("origin"), j.find("axis"), j.find("limit")
            self.joints[j.get("name")] = Joint(
                name=j.get("name"),
                type=j.get("type"),
                parent=j.find("parent").get("link"),
                child=j.find("child").get("link"),
                xyz=_floats(origin, "xyz"),
                rpy=_floats(origin, "rpy"),
                axis=_floats(axis, "xyz", default=(1.0, 0.0, 0.0)),
                lower=_optional_float(limit, "lower"),
                upper=_optional_float(limit, "upper"),
            )
        self.joint_into: dict[str, Joint] = {j.child: j for j in self.joints.values()}
        links = {lk.get("name") for lk in root.findall("link")}
        roots = links - set(self.joint_into)
        self.root = sorted(roots)[0] if roots else None

    def chain(self, link: str) -> list[Joint]:
        """Joints from the root down to `link`."""
        if link not in self.joint_into and link != self.root:
            raise KeyError(f"unknown link {link!r}")
        out = []
        while link in self.joint_into:
            joint = self.joint_into[link]
            out.append(joint)
            link = joint.parent
        return out[::-1]

    def joint_transform(
        self, joint: Joint, q: float = 0.0, perturb: dict[str, float] | None = None
    ) -> np.ndarray:
        """X_parent_child for `joint` at position `q`, with any perturbations of that joint applied."""
        p = perturb or {}
        name = joint.name
        xyz = joint.xyz + np.array([p.get(f"pos:{name}:{a}", 0.0) for a in ("x", "y", "z")])
        rot = rotation_from_rpy(*joint.rpy)
        extra = [p.get(f"rot:{name}:{a}", 0.0) for a in ("roll", "pitch", "yaw")]
        if any(extra):
            rot = rot @ rotation_from_rpy(*extra)
        x = pose(rot, xyz)
        if joint.type in ("revolute", "continuous"):
            angle = q + p.get(f"zero:{name}", 0.0)
            x = x @ pose(rotation_about_axis(joint.axis, angle), np.zeros(3))
        elif joint.type == "prismatic":
            x = x @ pose(
                np.eye(3), joint.axis / np.linalg.norm(joint.axis) * (q + p.get(f"zero:{name}", 0.0))
            )
        return x

    def link_pose(
        self, link: str, q: dict[str, float] | None = None, perturb: dict[str, float] | None = None
    ) -> np.ndarray:
        """X_root_link. `q` maps joint names to positions (missing joints are at 0)."""
        q = q or {}
        x = np.eye(4)
        for joint in self.chain(link):
            x = x @ self.joint_transform(joint, q.get(joint.name, 0.0), perturb)
        return x

    def point(self, link: str, p_link: np.ndarray, q=None, perturb=None) -> np.ndarray:
        """A point given in `link`'s frame, expressed in the root frame."""
        return (self.link_pose(link, q, perturb) @ np.append(np.asarray(p_link, dtype=float), 1.0))[:3]

    def moving_joints(self, link: str) -> list[str]:
        """Names of the joints with a position between the root and `link`."""
        return [j.name for j in self.chain(link) if j.moves]


@dataclass(frozen=True)
class PinholeCamera:
    """A camera rigidly attached to a URDF link.

    The link's x axis is the viewing direction (URDF/ROS body convention); the optical frame used for
    projection has z forward, x right, y down.
    """

    link: str
    fx: float
    fy: float
    ppx: float
    ppy: float
    width: int
    height: int

    # X_link_optical: optical z = link x, optical x = -link y (right), optical y = -link z (down)
    LINK_TO_OPTICAL = pose(np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]), np.zeros(3))

    def pose_in_root(self, tree: UrdfTree, q=None, perturb=None) -> np.ndarray:
        """X_root_optical."""
        return tree.link_pose(self.link, q, perturb) @ self.LINK_TO_OPTICAL

    def project(self, x_root_optical: np.ndarray, points_root: np.ndarray) -> np.ndarray:
        """Pixels (N, 2) of root-frame points (N, 3); NaN for points at or behind the camera."""
        pts = np.atleast_2d(points_root)
        homog = np.c_[pts, np.ones(len(pts))]
        p_c = (invert(x_root_optical) @ homog.T).T[:, :3]
        z = np.where(p_c[:, 2] > 1e-6, p_c[:, 2], np.nan)
        return np.stack([self.fx * p_c[:, 0] / z + self.ppx, self.fy * p_c[:, 1] / z + self.ppy], axis=1)

    def in_view(self, uv: np.ndarray) -> np.ndarray:
        """True where pixels `uv` (..., 2) fall inside the image (NaN is outside)."""
        u, v = uv[..., 0], uv[..., 1]
        with np.errstate(invalid="ignore"):
            return (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)


def _floats(element, attr: str, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    if element is None or element.get(attr) is None:
        return np.array(default, dtype=float)
    return np.array([float(v) for v in element.get(attr).split()], dtype=float)


def _optional_float(element, attr: str) -> float | None:
    if element is None or element.get(attr) is None:
        return None
    return float(element.get(attr))
