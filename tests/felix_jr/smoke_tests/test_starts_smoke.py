"""The grasp environment's starts (`sim2real.starts`) from a one-anchor bank: every pool, the attack's perturbation, the
check, and a bank from other servos refused."""

import json
from dataclasses import asdict

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("pinocchio")

from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel, pinch_frame  # noqa: E402
from lerobot.felix_jr.harness import api  # noqa: E402
from lerobot.felix_jr.sim2real import bank, starts  # noqa: E402
from lerobot.felix_jr.sim2real.assets import is_cached  # noqa: E402

pytestmark = pytest.mark.skipif(not is_cached(), reason="the SO-101 model is not cached")
ANCHORED = 4  # (as `test_bank_smoke.py`)


@pytest.fixture(scope="module")
def model():
    return XleRobotModel.load(api.MODEL)


@pytest.fixture(scope="module")
def bank_path(model, tmp_path_factory):
    config = bank.BankConfig()
    placement = bank.placements(model, config, ANCHORED + 1)[ANCHORED]
    line = bank.run_placement(model, config, placement)
    assert line["anchor"] is not None
    path = tmp_path_factory.mktemp("bank") / "bank.jsonl"
    head = {"kind": "config", "model": str(api.MODEL), "servos": bank.servos(), "config": asdict(config)}
    path.write_text(json.dumps(head) + "\n" + json.dumps(line) + "\n")
    return path


@pytest.mark.parametrize("pool", starts.POOLS)
def test_every_pool_gives_a_clean_start_with_the_object_where_the_anchor_had_it(model, bank_path, pool):
    sampler = starts.StartSampler(model, bank_path)
    start = sampler.sample(np.random.default_rng(0), pool)
    m, d = start.scene.m, start.scene.d
    assert start.pool == pool and start.anchor == 0 and start.draws <= 3
    obj = d.xpos[m.body("object").id]
    p = sampler.anchors[0]["placement"]
    assert np.linalg.norm(obj[:2] - p["bottom_root"][:2]) < 0.02
    if pool == "lifted":
        assert obj[2] > p["bottom_root"][2] + p["height"] / 2 + 0.002  # (partway up)


def test_the_attack_start_is_the_anchors_attack_offset_as_drawn(model, bank_path):
    """Without the execution's spread the gripper lands where the perception's offset puts it (the Jacobian's map,
    good to a few mm at these sizes, plus the servos' stop); the offsets' spread follows the config's widths."""
    sampler = starts.StartSampler(model, bank_path, starts.StartConfig(execution_width=0.0))
    anchor = sampler.anchors[0]["anchor"]
    q0 = model.joint_vector({"right": anchor["attack"]["joints"]})
    p0 = model.frame_pose(pinch_frame("right"), q0)
    rng = np.random.default_rng(1)
    misses, along = [], []
    for _ in range(12):
        start = sampler.sample(rng, "attack")
        o = start.offset_m
        want = p0[:3, :3] @ [o["across"], 0.0, o["along"]] + np.array([0.0, 0.0, o["up"]])
        commanded = {f"{j}.pos": v for j, v in start.joints_deg.items()}
        got = (
            model.frame_pose(pinch_frame("right"), model.joint_vector({"right": commanded}))[:3, 3]
            - p0[:3, 3]
        )
        misses.append(np.linalg.norm(got - want))
        along.append(o["along"])
    assert max(misses) < 0.005
    assert np.mean(along) < 0.0  # (biased short, as the sessions' attacks were)


def test_an_attack_with_the_arm_in_the_object_is_refused(model, bank_path):
    sampler = starts.StartSampler(model, bank_path)
    scene = sampler.scene(0)
    anchor = sampler.anchors[0]["anchor"]
    jaws = starts.open_gripper_pos(model, "right", anchor)
    scene.set_pose({**anchor["attack"]["joints"], "gripper.pos": jaws})
    assert starts.check(scene, "attack")
    scene = sampler.scene(0)
    # the jaws part closed through the object (16 mm in; shut, the pad has passed right through, and MuJoCo reports
    # 0.1 mm: an attack's offsets, 1-2 cm from 5 cm back, cannot put a pad all the way through)
    scene.set_pose({**anchor["grasp"]["joints"], "gripper.pos": 10.0})
    assert not starts.check(scene, "attack")


def test_a_bank_from_other_servos_or_no_pool_weight_is_refused(model, bank_path, tmp_path):
    lines = bank_path.read_text().splitlines()
    head = json.loads(lines[0])
    head["servos"] = {"kp": None, "frictionloss": None}
    other = tmp_path / "other.jsonl"
    other.write_text("\n".join([json.dumps(head), *lines[1:]]) + "\n")
    with pytest.raises(ValueError, match="other servos"):
        starts.StartSampler(model, other)
    with pytest.raises(ValueError, match="no pool"):
        starts.StartSampler(model, bank_path, starts.StartConfig(pool_weights={"attack": 0.0}))


def test_a_video_of_an_attack_pair_and_a_pool_clip_is_written(model, bank_path, tmp_path, monkeypatch):
    """The unperturbed anchor holds; the clip's frames are the pair's size (the single clip padded)."""
    cv2 = pytest.importorskip("cv2")
    monkeypatch.setattr(
        starts, "SCRIPT_S", {k: 0.6 if k != "start" else 0.1 for k in starts.SCRIPT_S} | {"lift": 1.0}
    )
    sampler = starts.StartSampler(model, bank_path)
    out = tmp_path / "starts.mp4"
    lines = starts.video(sampler, out, np.random.default_rng(0), pairs=1, clips={"in_jaws": 1})
    assert len(lines) == 2 and lines[0].startswith("1. attack pair") and ": held" in lines[0]
    assert lines[1].startswith("2. in_jaws")
    capture = cv2.VideoCapture(str(out))
    assert capture.get(cv2.CAP_PROP_FRAME_COUNT) > 30
    ok, frame = capture.read()
    assert ok and frame.shape == (2 * starts.TILE[1], 2 * starts.TILE[0], 3)
