"""`sim2real.refine` end to end: the greedy grasp's samples played in the simulator, a better grasp found."""

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("pinocchio")

from lerobot.felix_jr.calibration.joint_sim import REST  # noqa: E402
from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel  # noqa: E402
from lerobot.felix_jr.harness import api  # noqa: E402
from lerobot.felix_jr.sim2real import bank  # noqa: E402
from lerobot.felix_jr.sim2real.assets import is_cached  # noqa: E402
from lerobot.felix_jr.sim2real.refine import RefineSettings, refine  # noqa: E402

pytestmark = pytest.mark.skipif(not is_cached(), reason="the SO-101 model is not cached")


def test_the_refine_beats_the_greedy_grasp_and_returns_what_execute_runs():
    """Seed 0's placement 4: its greedy grasp (depth 0) lifts in 3 of 8 draws on the robot's misses (2026-11-19);
    two batches of four, four draws each, in two processes: a sample at least as good, its execution ready."""
    model = XleRobotModel.load(api.MODEL)
    placement = bank.placements(model, bank.BankConfig(), 5)[4]
    settings = RefineSettings(
        batch=4,
        max_batches=2,
        draws=4,
        accept=1.0,
        workers=2,
        table_z=placement.bottom_root[2],
        mass_kg=placement.mass_kg,
    )
    plan = refine(
        model,
        "right",
        placement.primitive(),
        {"t": 0.0986, "phi": 0.0, "psi": 30.0},
        1,
        {**REST, "gripper.pos": 2.0},
        bank.BankConfig().settings(),
        settings,
    )
    start = plan.samples[0]
    assert start.params["depth"] == 0.0 and not start.refused and len(start.rises) == 4
    assert plan.best.score >= start.score and plan.best.lifted > 0
    assert plan.execution is not None and plan.candidate is not None
    assert [m.label.split(" ")[0] for m in plan.execution.moves][:2] == ["start:", "stage"]
    assert np.isclose(plan.candidate.claw.depth, plan.best.params["depth"])


def test_the_harness_tool_refines_a_chosen_candidate_into_a_plan_execute_runs():
    """`refine_grasp` through the harness's runner: the greedy candidate as `grasp_candidates` would give it (one, named
    A), refined from the head; recorded, its execution the whole pick from the candidates' capture, its poses the
    pick's (to stop at the attack point with `select_poses`)."""
    from dataclasses import replace
    from types import SimpleNamespace

    from lerobot.felix_jr.harness.grasping import GraspCandidates, first_feasible

    model = XleRobotModel.load(api.MODEL)
    placement = bank.placements(model, bank.BankConfig(), 5)[4]
    target = placement.primitive()
    capture = SimpleNamespace(arms={"right": {**REST, "gripper.pos": 2.0}}, head_ticks=None)
    config = bank.BankConfig()
    found = first_feasible(
        target,
        model,
        capture,
        np.random.default_rng(0),
        arms=("right",),
        where={"psi": (0.0, 30.0)},
        settings=config.settings(),
        depths=(0.0,),
        max_tries=3000,
    )
    chosen = GraspCandidates((replace(found.candidate, name="A"),), 1, 1, (), {}, config.settings(), capture)
    h = api.Harness(model=model, policy=api.UNATTENDED)
    plan = h.call("refine_grasp", target, chosen, mass_kg=placement.mass_kg, max_batches=1, workers=3)
    assert h.ws.id_of(plan) is not None and len(plan.samples) == 6 and plan.batches == 1
    assert plan.execution is not None and plan.execution.poses.capture is capture
    assert list(plan.execution.poses.labels) == ["stage", "pick.back", "pick", "lift"]
    turned = (plan.best.params["phi"] - found.candidate.claw.grasp.params["phi"] + 180.0) % 360.0 - 180.0
    assert abs(turned) < 10.0  # (near the chosen side: phi wraps)
