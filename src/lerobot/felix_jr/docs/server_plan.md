# Robot server plan

How felix_jr is split between the Raspberry Pi on the robot and the computer experiments are written on, how the
two talk, and how code gets from one to the other.

## Goals

1. **Slow, deliberate control is fine.** The robot can stop after each sub-action and wait while the next one is
   planned. Nothing needs a real-time loop.
2. **Don't flood the network with images.** Images are sent only when asked for, never streamed.
3. **The target experiments test the native planning abilities of VLMs**, mostly pick and place.

## Physical setup

- The RealSense D415, both arms, the head and the wheels are all connected to a Raspberry Pi running Ubuntu.
- The servos have their own power supply. The RealSense is powered from the Pi's USB.
- The Pi has a clone of this repo and runs the server from it.
- Experiments are written and run on another computer (the "laptop") that connects to the Pi over the network.

## Architecture

The Pi runs a thin hardware server: it owns the hardware and enforces safety, but it does not plan. The laptop
does everything else. One step of an experiment is request/response:

```
laptop                                   Pi
------                                   --
capture()            ----------------->  latest RGB-D frame + joint state
render, ask the VLM, plan, run IK
move_joints(...)     ----------------->  run trajectory, return when settled
capture()            ----------------->  ...
```

### Pi: hardware server

- Keeps the RealSense pipeline running and holds only the latest frame. Auto-exposure needs a warm-up, so starting
  the camera per request would be slow.
- Runs motions with the existing `home.move_all` (trapezoid profiles, settle, `max_relative_target`).
- Enforces joint limits, a speed cap and a watchdog. If the client disconnects during a motion, it stops and holds.
- Accepts only joint-space commands. Inverse kinematics stays on the laptop, so the server only checks things that
  are easy to validate. It also avoids installing placo on aarch64.

| Command | What it does |
|---|---|
| `hello` | Returns the server's git commit, protocol version and which arms, head and base are present. |
| `get_state` | Returns joint positions, gripper openings and whether a motion is running. |
| `capture` | Returns one timestamped snapshot: color, depth, camera intrinsics/extrinsics and the joint state at the same moment. |
| `move_joints(arm, waypoints, speed)` | Runs a joint trajectory and returns when it has settled, or on a fault. |
| `gripper(arm, opening)` | Opens or closes a gripper. |
| `head(pan, tilt)` | Points the head. |
| `base(distance or angle)` | Drives or turns the base by a fixed amount. |
| `home`, `relax` | Moves to the home pose, or turns torque off. |
| `stop` | Stops any motion and holds. Must work while a `move_joints` is running. |

### Laptop: client and experiments

- A `FelixClient` class with blocking methods, so an experiment reads as a plain script:

  ```python
  snap = robot.capture()
  plan = ask_vlm(render(snap), task)
  robot.move_joints("left", ik(plan.waypoints))
  ```

- IK (placo), the scene tools (`depth.py`, `scene.py`, `overlay.py`), projecting trajectories into the image, and
  the VLM calls all run here.
- **The episode log is written on the laptop.** For each step it saves the snapshot, the rendered prompt image,
  the VLM's raw response, the commanded joints and the result. This log is the experiment data, so it lives with the
  experiment code.

## Communication

**ZeroMQ, ROUTER on the server and DEALER on the client, with msgpack messages.**

- `pyzmq` is already a LeRobot dependency (the LeKiwi extra).
- With ROUTER, the server can answer `stop` and `get_state` while a `move_joints` is still running. To allow this,
  motions run in a worker thread. Plain REQ/REP can't do this, and it locks up if a reply is lost.
- Images are sent as binary frames: color as JPEG or PNG, depth as a lossless 16-bit PNG. That is roughly
  0.3–0.5 MB per snapshot, which is small for Wi-Fi at a few snapshots per sub-action.
- LeKiwi's host/client is not a good template: it streams observations and actions at a fixed rate, which goal 2
  rules out.
- HTTP (FastAPI) would also work and is easier to poke at with `curl`, but a mid-motion `stop` needs more care.

## Development workflow

- **Laptop-only code** (experiments, prompts, planning) never needs deploying.
- **Server code:** commit and push, then from the laptop:

  ```bash
  ssh pi 'cd ~/lerobot && git pull --ff-only && systemctl --user restart felix-server'
  ```

  `rsync` the working tree instead to iterate without committing. VS Code Remote-SSH also works for editing on the
  Pi directly.
- **No "pull from GitHub" server command.** It would add a remote code-update path to the server and can leave it
  half-updated.
- **The server runs under systemd** (tmux is fine at first), so it starts at boot and restarts after a crash.
- **`hello` reports the server's git commit and protocol version**, and the client warns when they differ from
  the laptop's.
- **A fake-hardware mode** for the server, reusing the fake buses from `tests/felix_jr/`, lets the client and
  experiment code be developed on the laptop with no robot attached.

## Pi setup notes

- **pyrealsense2 is built from source** in `/home/zcecere/CodeRepository/librealsense` (bindings under `build/`).
  The server finds the module itself: `--librealsense-build` (default `/home/zcecere/CodeRepository/librealsense/build`)
  is searched for `pyrealsense2*.so` and that folder is added to `sys.path`. The server also opens the camera
  with `pyrealsense2` directly instead of LeRobot's `RealSenseCamera`, because a source build has no pip
  metadata, so LeRobot's availability check reports it as not installed and the camera class refuses to run.
  The Python version of the project's venv has to match the one the bindings were built for.
- **USB power.** The servos have their own supply, so the Pi's USB only powers the D415 and the two servo
  controllers' logic. If the camera drops out under load, try a powered USB hub.
- **Calibration has to be redone on the Pi.** That covers arm calibration, home poses, wheel signs and head signs
  (`~/.cache/huggingface/lerobot/calibration/robots/so_follower/`). Follow the first-time order in `README.md`.
  Serial port names will also differ, so add a Pi section to `current_ports.md`.

## Status

Built (`felix_jr/server/`, tests in `tests/felix_jr/test_server.py`): `protocol.py` (wire format), `hardware.py`
(`FakeHardware`, `RealHardware`), `server.py` (`hello`, `get_state`, `capture`) and `client.py` (`FelixClient`).

```bash
# on the Pi (add --fake to run with no hardware)
uv run python -m lerobot.felix_jr.server.server --left-port <PORT> --right-port <PORT>
# on the laptop
python -c "from lerobot.felix_jr.server.client import FelixClient; c = FelixClient('tcp://<pi>:5555'); print(c.hello())"
```

## First steps

1. ~~A minimal server with `hello`, `get_state` and `capture`, plus the matching client, tested against fake
   hardware on the laptop.~~ Done.
2. Run it on the Pi with the real camera, and check snapshot size and round-trip time over Wi-Fi.
3. Add `move_joints`, `stop` and the watchdog, and test `stop` during a motion.
4. Add `gripper`, `head`, `base`, `home` and `relax`.
5. Add the episode logger on the laptop, then the first pick-and-place experiment.
