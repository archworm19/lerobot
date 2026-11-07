"""The object-centric grasp tools as the model calls them (`harness.api`): fit, candidates, render, plan, refine, on a
rendered scene through `Harness.call` (`docs/object_grasping.md`, phase C)."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("scipy")

from test_fit_cylinder import EYE, JAR, marks_around, render  # noqa: E402

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel  # noqa: E402
from lerobot.felix_jr.harness import api  # noqa: E402
from lerobot.felix_jr.harness.grasping import GraspCandidates  # noqa: E402
from lerobot.felix_jr.harness.objects import Capture, Object, Poses  # noqa: E402

URDF = Path(__file__).parent / "reference" / "pen1_calibration_10312026.urdf"
REST = {  # bowl1's robot at rest
    "right": {
        "shoulder_pan.pos": -0.132,
        "shoulder_lift.pos": -102.989,
        "elbow_flex.pos": 96.527,
        "wrist_flex.pos": 71.341,
        "wrist_roll.pos": -0.220,
        "gripper.pos": 1.037,
    },
}
BOWL1_GRASP = {"t": (0.069, 0.069), "phi": (30.0, 60.0), "psi": (45.0, 45.0)}  # around the grasp that worked


@pytest.fixture(scope="module")
def model():
    return XleRobotModel.load(URDF)


@pytest.fixture
def setup(model):
    h = api.Harness(policy=api.UNATTENDED, model=model)
    rendered = render([JAR], EYE, JAR.bottom_root)
    capture = Capture(rendered.scene, rendered.x_root_camera, REST, {}, 0.0)
    h.ws.add(capture, "capture")
    view = h.call("full_view", capture)
    return h, capture, view


def test_the_grasp_tools_are_software_and_documented():
    for name in ("fit_cylinder", "grasp_candidates", "plan_pick", "refine"):
        spec = api.TOOLS[name]
        assert spec.effect is api.Effect.SOFTWARE and spec.fn.__doc__
        assert name in api.describe_tools()


def test_fit_choose_draw_plan_refine(setup):
    h, capture, view = setup
    jar = h.call("fit_cylinder", view, marks_around(capture, JAR), "up", "jar")
    assert isinstance(jar, Object) and jar.part().radius == pytest.approx(JAR.radius, abs=0.002)
    found = h.call("grasp_candidates", jar, capture, arms=["right"], where=BOWL1_GRASP)
    assert isinstance(found, GraspCandidates) and found.passed > 0 and found.label == "jar"
    assert found.capture is capture and "settings:" in found.describe()
    image = h.call("render", view, jar, found)
    assert image.shape[:2] == (capture.scene.height, capture.scene.width)
    pick = h.call("plan_pick", found, "A")
    assert isinstance(pick, Poses) and pick.labels == ("stage", "pick.back", "pick", "lift")
    hold = h.call("select_poses", pick, "stage", "pick.back")
    assert hold.labels == ("stage", "pick.back")
    again = h.call("refine", found, "A", jar, capture)  # nothing moved, nothing missed: the same grasp
    assert again.passed == 1
    np.testing.assert_allclose(again["A"].claw.root_from_pinch, found["A"].claw.root_from_pinch, atol=1e-9)
    assert [c["tool"] for c in h.calls][-6:] == [
        "fit_cylinder",
        "grasp_candidates",
        "render",
        "plan_pick",
        "select_poses",
        "refine",
    ]


def test_settings_reach_the_tool_and_bad_ones_are_named(setup):
    h, capture, view = setup
    jar = h.call("fit_cylinder", view, marks_around(capture, JAR), "up", "jar")
    narrow = h.call(
        "grasp_candidates",
        jar,
        capture,
        arms=["right"],
        where=BOWL1_GRASP,
        rotation_bounds_deg=[45, 0.01, 45],
    )
    assert narrow.settings.rotation_bounds_deg == (45.0, 0.01, 45.0)
    with pytest.raises(ValueError, match="unknown settings"):
        h.call("grasp_candidates", jar, capture, arms=["right"], where=BOWL1_GRASP, friction=0.4)


def test_stale_captures_are_refused(setup):
    h, capture, view = setup
    jar = h.call("fit_cylinder", view, marks_around(capture, JAR), "up", "jar")
    found = h.call("grasp_candidates", jar, capture, arms=["right"], where=BOWL1_GRASP)
    h.ws.moved()  # the arm moved: everything seen before is stale
    with pytest.raises(ValueError, match="before the arm last moved"):
        h.call("grasp_candidates", jar, capture, arms=["right"], where=BOWL1_GRASP)
    with pytest.raises(ValueError, match="before the arm last moved"):
        h.call("plan_pick", found, "A")
    with pytest.raises(ValueError, match="before the arm last moved"):
        h.call("refine", found, "A", jar, capture)
    # at the hold: a fresh capture; the refined grasp descends from the old candidates but plans (its capture is new)
    fresh = Capture(capture.scene, capture.x_root_camera, REST, {}, 1.0)
    h.ws.add(fresh, "look")
    jar2 = h.call("fit_cylinder", h.call("full_view", fresh), marks_around(fresh, JAR), "up", "jar")
    again = h.call("refine", found, "A", jar2, fresh, None, None, see_at_grasp=False)
    assert again.passed == 1 and again.settings.see_at_grasp is False
    plan = h.call("plan_pick", again, "A")
    assert plan.labels[1] == "pick.back"
    go_in = h.call("select_poses", plan, "pick.back", "pick")
    with pytest.raises(
        ValueError, match="no gripper map|gripper"
    ):  # past the freshness check: the test model has no map
        h.call("prepare_execution", go_in, {"pick": "close"})
    with pytest.raises(ValueError, match="before the arm last moved"):  # the old plan's capture is stale
        h.call("prepare_execution", h.call("select_poses", h.call("plan_pick", found, "A"), "stage"))


def test_the_tools_it_replaces_are_marked_deprecated(setup):
    """`docs/object_grasping.md`, section 4: the model is told what replaces them; a call to one is logged as such."""
    documented = api.describe_tools()
    for name in ("solve_pose", "approach", "join_poses", "held_object"):
        assert api.TOOLS[name].deprecated
        block = documented[documented.index(f"{name}(") :]
        assert block.split("\n")[1].startswith("DEPRECATED: ")
    assert not api.TOOLS["select_poses"].deprecated  # stays, for staging
    h, capture, view = setup
    h.call("full_view", capture)
    assert "deprecated" not in h.calls[-1]
    first = h.call("solve_pose", "right", capture, [-0.33, -0.08, 0.81], -40.0, label="a")
    second = h.call("solve_pose", "right", capture, [-0.33, -0.08, 0.84], -40.0, label="b")
    h.call("join_poses", first, second)
    assert h.calls[-1]["deprecated"].startswith("plan_pick gives")


def test_a_move_refused_before_it_starts_leaves_what_was_seen_fresh(setup):
    """bowl2: an `execute` refused for a stale execution marked the fresh look stale too, though nothing moved."""
    h, capture, view = setup
    epoch = h.ws.epoch

    def refuse(harness):
        raise api.NotStartedError("stale: nothing moved")

    def moved_then_failed(harness):
        raise RuntimeError("stopped halfway")

    api.TOOLS["_refuse"] = api.Tool("_refuse", refuse, api.Effect.MOVE_ARM, False)
    api.TOOLS["_halfway"] = api.Tool("_halfway", moved_then_failed, api.Effect.MOVE_ARM, False)
    try:
        with pytest.raises(api.NotStartedError):
            h.call("_refuse")
        assert h.ws.epoch == epoch and not h.ws.is_stale(capture)
        with pytest.raises(RuntimeError):
            h.call("_halfway")
        assert h.ws.epoch > epoch and h.ws.is_stale(capture)  # a move that failed may have moved: stale
    finally:
        del api.TOOLS["_refuse"], api.TOOLS["_halfway"]
