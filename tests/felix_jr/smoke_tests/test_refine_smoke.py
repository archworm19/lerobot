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
