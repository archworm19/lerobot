"""Object-centric grasping, phase B's replay of pen2's book scene (`docs/object_grasping.md`, `docs/sessions.md`).

pen2 (2026-11-05), after the reset: a pen standing vertical in the top edge of an upright book (the pen at (-45.5,
-2.9) cm, its top at 97.4 cm, the book's top edge at 92.5 cm). The left arm took it at pitch 0, heading 163, roll
180, the grasp at 95.8 cm: it reached the pen cleanly (the jaws and the pen in view at the attack point; the lift
lost the pen to a weak grip, which this does not model). The re-grasp went deeper (more of the gripper over the book's
edge) and the gripper hit the book on the way in. Checked, with no robot, as the model would ask for it: the pen described only where it may be grasped (above the
book), the approach level or from above (the tool knows nothing of the book, so the model says from where):

1. The session's first left grasp is among the best: pitch 0, heading 163, roll 180.
2. Every best candidate approaches level or from above, and takes the pen above the book.

Unrestricted, the best come from below (pitch +26 to +51: through the book), nearly antipodal, joints near their
bounds: what the `where` above is for (`docs/object_grasping.md`, phase B progress).
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("scipy")

from lerobot.felix_jr.calibration.xle_robot_model import ArmBox, XleRobotModel  # noqa: E402
from lerobot.felix_jr.harness.grasping import grasp_candidates  # noqa: E402
from lerobot.felix_jr.harness.objects import Object, angles_of  # noqa: E402
from lerobot.felix_jr.harness.primitives import Cylinder  # noqa: E402

URDF = Path(__file__).parents[3] / "src" / "lerobot" / "felix_jr" / "urdfs" / "xle_calib_11062026.urdf"
# the left arm's parts as boxes: the bounds of the capsules fitted for that URDF (before the boxes; loading it now
# leaves its capsules out)
LEFT_BOXES = {
    "gripper_body": ArmBox("left_gripper_link", (0.0067, 0.008, -0.0307), (0.0657, 0.0487, 0.0389)),
    "forearm": ArmBox("left_lower_arm_link", (-0.0445, 0.004, 0.0075), (0.1496, 0.0788, 0.0752)),
}
CAPTURE = SimpleNamespace(  # pen2's capture21: the arms at rest, the head as it scanned
    arms={
        "left": {
            "shoulder_pan.pos": -7.209,
            "shoulder_lift.pos": -102.725,
            "elbow_flex.pos": 96.396,
            "wrist_flex.pos": 67.253,
            "wrist_roll.pos": -1.626,
            "gripper.pos": 0.975,
        },
        "right": {
            "shoulder_pan.pos": -0.835,
            "shoulder_lift.pos": -103.077,
            "elbow_flex.pos": 96.527,
            "wrist_flex.pos": 71.516,
            "wrist_roll.pos": -0.22,
            "gripper.pos": 1.037,
        },
    },
    head_ticks={"head_neck": 1480, "head_eyes": 1493},
)
BOOK_TOP_Z = 0.925
PEN_TOP_Z = 0.974
GRASPABLE_FROM_Z = 0.93  # the pen above the book: all the tool is told of it
PEN = Object(
    "pen",
    {
        "barrel": Cylinder.upright(
            np.array([-0.455, -0.029, GRASPABLE_FROM_Z]), PEN_TOP_Z - GRASPABLE_FROM_Z, 0.0045
        )
    },
)
WORKED = (0.0, 163.0, 180.0)  # pitch, heading, roll of the left arm's first grasp


def _wrap(deg: float) -> float:
    return abs((deg + 180.0) % 360.0 - 180.0)


def off_worked(solution) -> float:
    pitch, heading, roll = angles_of(solution.root_from_pinch[:3, :3], WORKED[1])
    return max(abs(pitch - WORKED[0]), _wrap(heading - WORKED[1]), _wrap(roll - WORKED[2]))


@pytest.fixture(scope="module")
def found():
    model = XleRobotModel.load(URDF).with_boxes("left", LEFT_BOXES)
    return grasp_candidates(PEN, model, CAPTURE, arms=("left",), where={"psi": (0.0, 90.0)}, n=10)


def test_the_first_left_grasp_is_among_the_best(found):
    assert any(off_worked(c.grasp) <= 4.0 for c in found.candidates), found.describe()


def test_the_best_come_level_or_from_above(found):
    assert found.passed > 0
    for c in found.candidates:
        pitch, _, _ = angles_of(c.grasp.root_from_pinch[:3, :3])
        assert pitch <= 2.0, c.describe()
        assert c.claw.grasp.center_root[2] > BOOK_TOP_Z
