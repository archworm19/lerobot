# felix_jr

Our own code for an XLeRobot (two SO-101 arms, a head with a RealSense, an optional wheel base), a robot server on
the Raspberry Pi, camera-to-robot calibration, and the VLM harness for pick and place. The repo-root `CLAUDE.md`
still applies; this file covers how we work in this directory.

## Read the docs first

`docs/` is the source of truth for setup, design and what has been tried. Check it before changing code or
running anything on the robot, and update it when what you do changes what it says.

- **Start with `docs/layout.md`**: what each folder is, which computer it runs on (the Pi or your computer), and
  where each kind of state lives: robot settings on the robot (`setup/`), the calibrated URDF on your computer
  (`calibration/`), recordings in `scratch/`.
- **Run books** (how to set up and run things):
  - `docs/README.md`: first-time setup (new arm, new computer), the server, head range, calibration, the gripper,
    checks, teleop, troubleshooting, tests.
  - `docs/current_ports.md`: serial ports of each arm on each computer.
  - `docs/realsense_m1.md`: the RealSense on the Mac (pitfalls, what is verified).
- **Protocols** (how to drive a session on the robot): `docs/protocols.md`. Follow them when running a
  session.
- **Design documents** (why things are the way they are):
  - `docs/harness.md`: the VLM harness (research summary, design principles 1.A-1.K, experiment plan).
  - `docs/object_grasping.md`: the object-centric grasping experiment (phase A built): plan, interfaces, frame naming, flow, what it replaces, open questions.
  - `docs/camera_calibration.md`: the click calibration and `XleRobotModel`.
  - `docs/server_plan.md`: the Pi / laptop split and the protocol.
  - `docs/head.md`: the head servos.
- **Experiments** (what was tried, with numbers):
  - `docs/sessions.md`: each harness session on the robot, newest first: setup, what happened, takeaways.
  - `docs/tuning_notes.md`: hardware tuning findings, marked *measured* / *inferred* / *open*.
  - `docs/open_issues.md`: what is known missing or wrong in the harness. Close an item by deleting it; add new
    ones at the end of their section. Other areas keep their own "not done" lists (linked from its header).

## Running things

Everything runs as a module from the repo root:

```bash
uv run python -m lerobot.felix_jr.<folder>.<script> ...
```

Extras: `--extra feetech` (servos), `--extra pyzmq-dep` (the server), `--extra felix_jr` (pinocchio, placo,
scipy: calibration and reach). Anything that moves the robot needs a person watching; scripts that move it ask
first (`api.SAFE`) and go to rest on exit. Rehearse against the fake robot (`server.server --fake`) when you can.

New scripts go in `setup/` if what they write is a robot setting, and in `calibration/` if it goes into the calibrated
URDF (`docs/layout.md`, "setup/ and calibration/"). The URDF records the robot settings it was made under, and
`XleRobotModel.check_robot` refuses a robot whose settings changed since; when something new depends on a robot
setting, record it in the URDF and check it the same way.

## Testing

Tests are in `tests/felix_jr/` and never need hardware: the server and calibration tests run a fake robot behind
a real socket, and `tests/felix_jr/conftest.py` has a synthetic camera scene.

- **Fast tests**: run these after every change (~25 s):
  `uv run pytest tests/felix_jr -q -m "not smoke and not slow"`
- **Slow / smoke tests**: run before calling a change done when it touches the server, motion, calibration or
  reach, or the whole suite at the end of a larger change:
  `uv run pytest tests/felix_jr -q`
  - `smoke`: everything in `tests/felix_jr/smoke_tests/` (marked automatically by its `conftest.py`): end to end,
    e.g. a whole calibration session, IK on the experiments' waypoints, a server wiggle.
  - `slow`: unit tests that wait on real timeouts (watchdogs, heartbeats), marked `@pytest.mark.slow`.

New logic gets a fast unit test. A new end-to-end flow gets a smoke test in `smoke_tests/`; a unit test that has to
wait on a real timeout gets `@pytest.mark.slow`. Tests that need `--extra felix_jr` use
`pytest.importorskip("pinocchio")` (etc.) so the fast suite still runs without it. Frozen reference data lives in
`tests/felix_jr/reference/`.

## Dependencies

Adding a new dependency is fine, but **ask the user first**. Prefer an existing extra (`felix_jr`, `feetech`,
`pyzmq-dep`, `intelrealsense`) and add it to `pyproject.toml` rather than `uv pip install`-ing it (a later
`uv sync` removes untracked packages; see "Setup notes" in `docs/README.md`). Guard optional imports as the
root `CLAUDE.md` describes.

## Recorded data goes in scratch, never in git

Anything a run records (captures, renders, logs, pickled state, calibration clicks) goes in a `scratch/` folder,
not next to the code. `**/scratch/` is gitignored at the repo root, so nothing under it is committed, including
`harness/scratch/session.py` itself.

- Harness sessions write to `harness/scratch/sessions/<name>/` (`workspace.pkl`, `log.jsonl`, `<id>.png`).
  Name a session after its task plus a number (`pen1`, `cable1`), and keep old ones: later work refers to them.
- What a session or experiment *taught* is written up in `docs/sessions.md` / `docs/tuning_notes.md` with the
  numbers, since the raw data is not kept in git. Tests that need recorded data copy the few values they need
  into the test or `tests/felix_jr/reference/`.
- Do not `git add -f` anything under `scratch/`, and do not commit CSVs, images or captures elsewhere.
