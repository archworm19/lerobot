"""One placement through the bank end to end: the filters, the simulated grasp under physics draws, the anchor and its
snapshots, which load back into the same scene with the object in the jaws."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("pinocchio")

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel  # noqa: E402
from lerobot.felix_jr.harness import api  # noqa: E402
from lerobot.felix_jr.sim2real import bank  # noqa: E402
from lerobot.felix_jr.sim2real.assets import is_cached  # noqa: E402
from lerobot.felix_jr.sim2real.grasp_check import SNAPSHOTS  # noqa: E402
from lerobot.felix_jr.sim2real.scene import Cylinder, build  # noqa: E402

pytestmark = pytest.mark.skipif(not is_cached(), reason="the SO-101 model is not cached")
ANCHORED = 4  # seed 0's placement 4 (43 mm x 158 mm, 54 g) anchors at its first simulated grasp (2026-11-19)


def test_a_placement_is_anchored_and_its_snapshots_restore_the_grip():
    model = XleRobotModel.load(api.MODEL)
    config = bank.BankConfig()
    placement = bank.placements(model, config, ANCHORED + 1)[ANCHORED]
    line = bank.run_placement(model, config, placement)
    anchor = line["anchor"]
    assert anchor is not None, line["rejected_by_stage"]
    assert anchor["rate"] >= config.min_rate and len(anchor["trials"]) == config.n_draws
    assert anchor in [
        g | {"states": anchor["states"], "states_draw": anchor["states_draw"]} for g in line["simulated"]
    ]
    assert {"attack", "grasp", "lift"} <= set(anchor) and anchor["lift"]["joints"]
    assert set(anchor["states"]) == set(SNAPSHOTS)
    # the in-jaws state, into a scene built as the bank's (the nominal physics), held there a second: the object stays
    bottom = placement.bottom_root
    obj = Cylinder("object", bottom[0], bottom[1], placement.radius, placement.height, placement.mass_kg)
    scene = build(model, config.arm, [obj], table_z=bottom[2], finger_model=config.finger_model)
    m, d = scene.m, scene.d
    state = anchor["states"]["in_jaws"]
    assert len(state["qpos"]) == m.nq and len(state["ctrl"]) == m.nu
    d.qpos[:], d.qvel[:], d.ctrl[:] = state["qpos"], state["qvel"], state["ctrl"]
    mujoco.mj_forward(m, d)
    start = d.xpos[m.body("object").id].copy()
    for _ in range(int(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    assert np.linalg.norm(d.xpos[m.body("object").id] - start) < 0.005
    lifted = np.array(anchor["states"]["lifted"]["qpos"])
    jaws = np.array(state["qpos"])
    adr = m.joint(m.body("object").jntadr[0]).qposadr[0]
    assert lifted[adr + 2] > jaws[adr + 2]  # (partway up)
