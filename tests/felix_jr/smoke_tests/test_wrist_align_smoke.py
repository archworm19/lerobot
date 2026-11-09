"""Lining the jaws up from the wrist camera (`harness.wrist`, `docs/object_grasping.md` section 6), end to end against
the fake robot: wrist_look -> plan_alignment -> align_step, again and again.

The robot's joints read 1-2 deg off the truth (the arm's error the head camera's plan carries), and an upright cylinder
2 cm wide stands 6 cm ahead of the jaws, off the approach line. The two sides a person (the model) would mark are its
true silhouette's. The steps fall short (`STEP_FACTOR`) rather than overshoot, and the jaws end up on it: on the true
line, not on one a shortened depth would bias (the first version's flaw).
"""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("scipy")

from test_wrist_camera import TRUE_CAMERA, URDF, _with_pose, start_joints  # noqa: E402

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel, pinch_frame  # noqa: E402
from lerobot.felix_jr.harness import api  # noqa: E402
from lerobot.felix_jr.server.client import FelixClient  # noqa: E402
from lerobot.felix_jr.server.hardware import FakeHardware  # noqa: E402
from lerobot.felix_jr.server.server import RobotServer  # noqa: E402

ARM = "right"
WIDTH, MARGIN = 0.02, 0.008
JOINT_ERROR_DEG = {"shoulder_pan.pos": 1.5, "elbow_flex.pos": -1.0, "wrist_flex.pos": 2.0}  # truth - reading
OFFSET_M = (0.008, -0.005)  # the object off the approach line: toward the moving jaw, square to that
AHEAD_M = 0.06


@pytest.fixture(scope="module")
def model():
    return XleRobotModel.load(URDF).with_wrist_camera(ARM, _with_pose(TRUE_CAMERA))


@pytest.fixture
def client(tmp_path):
    fake = FakeHardware(arm_time_scale=0.01, head_range_path=tmp_path / "range.json")
    server = RobotServer(fake)
    address = server.bind("tcp://127.0.0.1:*")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with FelixClient(address, timeout_s=10.0) as c:
        yield c
    server.shutdown()
    thread.join(timeout=3.0)
    server.close()


def _unit(v):
    return v / np.linalg.norm(v)


def true_q(model, reading):
    return model.joint_vector({ARM: {k: v + JOINT_ERROR_DEG.get(k, 0.0) for k, v in reading.items()}})


@pytest.mark.parametrize(
    ("true_width", "advance", "within_m"),
    [
        (
            WIDTH,
            0.0,
            0.002,
        ),  # the plan's width right: on the line in a couple of steps, from the width's depth
        # the object 25% wider than the plan says: the width's depth 20% short, ~1 cm sideways at any range (the
        # first version settled there); a sidestep first, then the depth from the parallax, coming in half the way
        (1.25 * WIDTH, 0.5, 0.003),
    ],
)
def test_the_jaws_are_lined_up_by_conservative_steps(model, client, true_width, advance, within_m):
    """The steps fall short, never past; with a wrong width, the parallax of a sidestep gives the depth."""
    joints, _, _ = start_joints(model, ARM)
    start = {**joints, "gripper.pos": 30.0}
    with client.session():
        client.move_joints({f"{ARM}_arm": start}, tol=0.5)
        client.keep_holding()
    reading = client.get_state()[f"{ARM}_arm"]
    # the plan's jaw line (pinch frame): the aim point, the approach (the pinch frame's +z here)
    aim = model.grasp_offset(ARM, WIDTH, MARGIN)
    approach = np.array([0.0, 0.0, 1.0])
    toward = _unit(aim - (aim @ approach) * approach)
    square = _unit(np.cross(approach, toward))
    # the object, placed by the truth: off the jaws' line by OFFSET_M, AHEAD_M ahead; upright along the hinge
    true_pinch = model.frame_pose(pinch_frame(ARM), true_q(model, reading))
    center_pinch = aim + AHEAD_M * approach + OFFSET_M[0] * toward + OFFSET_M[1] * square
    center_root = true_pinch[:3, :3] @ center_pinch + true_pinch[:3, 3]
    axis_root = true_pinch[:3, :3] @ square

    def offset_now():
        """The object's true offset from the jaws' line (toward, square), m."""
        pinch = model.frame_pose(pinch_frame(ARM), true_q(model, client.get_state()[f"{ARM}_arm"]))
        local = pinch[:3, :3].T @ (center_root - pinch[:3, 3]) - aim
        return np.array([local @ toward, local @ square, local @ approach])

    def true_edges():
        """The cylinder's silhouette at the grasp height, seen by the true camera: what a person would mark."""
        camera = model.wrist_camera_pose(ARM, true_q(model, client.get_state()[f"{ARM}_arm"]))
        c = camera[:3, :3].T @ (center_root - camera[:3, 3])
        v = camera[:3, :3].T @ axis_root
        across = c - (c @ v) * v
        d = np.linalg.norm(across)
        e, w = across / d, np.cross(v, across / d)
        r = true_width / 2
        sides = [c - (r * r / d) * e + s * r * np.sqrt(1 - r * r / d / d) * w for s in (1, -1)]
        return TRUE_CAMERA.project(np.array(sides))

    pick = SimpleNamespace(  # what jaw_line reads off a plan_pick
        width=WIDTH,
        grasp=aim,
        labels=("pick.back", "pick"),
        points=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.05]]),
        frames=np.array([np.eye(4), np.eye(4)]),
    )
    h = api.Harness(ask=lambda spec, preview: True, client=client, model=model)
    offsets = [offset_now()]
    view = first_view = h.call("wrist_look", ARM)
    image = h.call("render_wrist", view, pick, true_edges())
    assert image.shape == view.color.shape
    gains, sources = [], []
    center = None
    sidestep = 0.015 if advance else 0.0
    if sidestep:  # first, aside for the parallax (and no closer: the depth is not known yet)
        alignment = h.call("plan_alignment", view, pick, true_edges(), gain=0.0, sidestep_m=sidestep)
        assert "the width only" in alignment.depth_source
        before = alignment.center_uv
        view = h.call("align_step", alignment)
        offsets = [offset_now()]
        after = h.call("plan_alignment", view, pick, true_edges()).center_uv
        center = h.call("triangulate_wrist", first_view, before, view, after)  # from before the step: allowed
        assert center.parallax_deg > 3 and center.miss_m < 0.001, center.describe()
        # in the model's root frame, which the joints' error shifts as a whole: its distance from the camera is what
        # counts, against the truth's
        seen = np.linalg.norm(center.xyz - view.root_from_camera[:3, 3])
        true_camera = model.wrist_camera_pose(ARM, true_q(model, client.get_state()[f"{ARM}_arm"]))
        assert abs(seen - np.linalg.norm(center_root - true_camera[:3, 3])) < 0.005, center.describe()  # 3%
    for i in range(4):
        # center first; come in once near the line (as the tool's docstring says)
        alignment = h.call(
            "plan_alignment", view, pick, true_edges(), advance=advance if i else 0.0, center=center
        )
        assert alignment.ok and not alignment.clipped, alignment.describe()
        sources.append(alignment.depth_source)
        if alignment.last_gain is not None:
            gains.append(alignment.last_gain)
        view = h.call("align_step", alignment)
        offsets.append(offset_now())
    across = [np.linalg.norm(o[:2]) for o in offsets]
    assert across[1] < 0.5 * across[0], across  # the first correction does most of it
    assert offsets[1][:2] @ offsets[0][:2] > 0  # and falls short: no overshoot
    assert across[-1] < within_m, across
    assert offsets[-1][2] > 0.0  # never past the aim point
    assert gains and all(0.8 < g < 1.2 for g in gains), gains  # the steps went as predicted
    if sidestep:
        assert all("triangulation" in source for source in sources), sources  # still good after the steps
    h.call("home")
    with pytest.raises(ValueError, match="before home"):  # another arm move: the old views no longer count
        h.call("triangulate_wrist", first_view, [320.0, 240.0], view, [330.0, 240.0])
    with pytest.raises(ValueError, match="before the arm last moved"):
        h.call("plan_alignment", view, pick, true_edges())


class ElbowShort:
    """The fake robot's client, with the elbow stopping 1.2 deg short of every target (bowl6's elbow under load)."""

    def __init__(self, client):
        self.client = client

    def __getattr__(self, name):
        return getattr(self.client, name)

    def move_joints(self, targets, **kwargs):
        key = f"{ARM}_arm"
        now = self.client.get_state()[key]["elbow_flex.pos"]
        target = targets[key]["elbow_flex.pos"]
        short = np.sign(target - now) * min(1.2, abs(target - now))
        return self.client.move_joints({key: {**targets[key], "elbow_flex.pos": target - short}}, **kwargs)


def test_a_joint_that_stops_short_is_pushed_again(model, client):
    joints, _, _ = start_joints(model, ARM)
    with client.session():
        client.move_joints({f"{ARM}_arm": {**joints, "gripper.pos": 30.0}}, tol=0.5)
        client.keep_holding()
    aim = model.grasp_offset(ARM, WIDTH, MARGIN)
    pick = SimpleNamespace(
        width=WIDTH,
        grasp=aim,
        labels=("pick.back", "pick"),
        points=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.05]]),
        frames=np.array([np.eye(4), np.eye(4)]),
    )
    h = api.Harness(ask=lambda spec, preview: True, client=ElbowShort(client), model=model)
    view = h.call("wrist_look", ARM)
    center = view.project_pinch((aim + [0.0, 0.0, 0.06])[None, :])[0]
    edges = [center - [30.0, 0.0], center + [30.0, 0.0]]
    alignment = h.call("plan_alignment", view, pick, edges, gain=0.0, sidestep_m=0.015)
    after = h.call("align_step", alignment)
    reached = client.get_state()[f"{ARM}_arm"]
    assert all(abs(reached[k] - v) <= 0.5 for k, v in alignment.joints.items()), (reached, alignment.joints)
    assert "pushed again" in after.describe() and "still short" not in after.describe(), after.describe()


def test_hold_views_then_a_planned_move(model, client):
    """The contacts' flow without the depth: wrist_hold_views rolls and comes back, plan_wrist_move -> align_step moves
    the pinch point by the move asked, its turn kept; the hold's views go stale."""
    joints, _, _ = start_joints(model, ARM)
    with client.session():
        client.move_joints({f"{ARM}_arm": {**joints, "gripper.pos": 30.0}}, tol=0.5)
        client.keep_holding()
    h = api.Harness(ask=lambda spec, preview: True, client=client, model=model)
    start = client.get_state()[f"{ARM}_arm"]
    hold = h.call("wrist_hold_views", ARM)
    assert hold.labels == ("start", "roll -15", "roll +15", "back") and hold.current == 3, hold.labels
    rolls = [v.arm_pose["wrist_roll.pos"] - start["wrist_roll.pos"] for v in hold.views]
    assert np.allclose(rolls, [0.0, -15.0, 15.0, 0.0], atol=0.5), rolls
    assert abs(rolls[hold.current]) < 0.5  # back where it was
    view = hold.views[hold.current]
    before = view.root_from_pinch
    move = np.array([0.01, -0.005, 0.01])
    planned = h.call("plan_wrist_move", hold, move_m=move)
    assert planned.ok and not planned.clipped, planned.describe()
    after = h.call("align_step", planned)
    reached = after.root_from_pinch
    assert np.linalg.norm(reached[:3, 3] - before[:3, 3] - move) < 0.002, reached[:3, 3] - before[:3, 3]
    assert np.allclose(reached[:3, :3], before[:3, :3], atol=0.02)
    with pytest.raises(ValueError, match="before the arm last moved"):
        h.call("plan_wrist_move", hold, move_m=move)
