# Head (neck + eyes)

Two servos on the left arm's bus carry the RealSense: the **eyes** tilt it up and down, the **neck** turns it
left and right. This page covers how they are identified and what is still missing.

## Servos and wiring

| Name | Servo ID | Does | Positive direction |
|---|---|---|---|
| `head_neck` | 7 | turns the head left / right | LEFT (counter-clockwise seen from above) |
| `head_eyes` | 8 | tilts the camera up / down | DOWN |

IDs 7 and 8 follow XLeRobot's numbering for the head; the arm uses 1 to 6 and the wheels 9 and 10 (on the right
arm's bus). Wiring: left arm's controller board -> eyes servo -> neck servo, with the arm on the same board.
LeRobot finds servos by ID on the bus, so the head servos only need unique IDs.

The positive directions match the joint conventions in `xlerobot.urdf` (checked by computing the camera
direction from the URDF's `head_pan_joint` and `head_tilt_joint`, not on the hardware): a positive pan turns
the camera counter-clockwise seen from above (left), and a positive tilt pitches it **down**. That is why the
script asks "did it move down?" for the eyes.

## Setting the IDs

```bash
uv run python -m lerobot.felix_jr.setup.head <LEFT_ARM_PORT> left_arm
```

New servos leave the factory as ID 1, which collides with the arm's `shoulder_pan` and with each other, so:

1. Unplug the arm from the board, and the neck from the eyes. The board then talks to the eyes alone, and the
   script gives them ID 8.
2. Plug the neck back into the eyes (the arm stays off). The script scans the chain for the one servo that is
   not the eyes and gives it ID 7. (It also scans for the eyes in step 1 and stops if anything else answers.)
3. Plug the arm back in.

After each ID the script moves that servo about 7 degrees (80 ticks) and back and asks which way it went, then
saves a sign per servo to `<id>_head.json` next to the arm's calibration: +1 if the servo's raw positive
direction is down (eyes) / left (neck), -1 if not. If a move would run past the end of the servo's range it goes
the other way and the sign accounts for that.

## What went wrong the first time, and the guard

The first version let LeRobot's stock `setup_motor` find the new servo by broadcast ping. With the arm still
plugged into the board, several servos answer that ping at once and the helper takes whichever comes first.
So it renamed one of the left arm's own servos to ID 8 and then nudged it, which moved the arm; running it a
second time did the same to another arm servo. Two servos then shared ID 8, which nothing can address (a ping of
that ID comes back "Incorrect status packet!"), and IDs 1 and 2 stopped answering.

Now `setup.head` and `setup.wheels` scan the chain first and **refuse to write anything** unless exactly one
new servo answers (besides the ones already set up in this run). If the arm is still connected the scan sees
its IDs and stops. Servos that already share an ID are invisible to the scan, so that stops too (nothing
found). The tests cover each of those refusals. If it happens anyway, `diagnostics/set_servo_id.py` changes a
single servo's ID and refuses unless the old ID answers cleanly and the new one is free. To separate two servos
that share an ID, disconnect all but one from the board first, rename that one, reconnect, and repeat.

## In code

`make_follower(port, id)` adds the head servos to the left arm's bus automatically once `<id>_head.json`
exists (`with_head=None` means "detect"; pass `True` / `False` to force it). Connecting then pings IDs 7 and 8
too, so a missing head fails at connect instead of later. The head servos are kept out of the arm interface:
`get_observation()`, `send_action()`, the calibration file and the home pose only ever see the six arm joints.
The arm-only tools (teleop, `lift_test`, `dump_servo_config`) pass `with_head=False`; the wiggle test takes
`--no-head`.

## Not done yet

- The eyes' tilt range is set by `setup.head_range` (see the README, section 4): up limit, down limit and home,
  saved as raw ticks in `<id>_head_range.json`. The neck uses `--name head_neck` (left / right arrows, `[` / `]` for the limits),
  and the servos' normalized "degrees" in `get_observation` are still a full-range placeholder measured from
  raw 2048. The URDF suggests about -43 deg (up) to +83 deg (down) for the tilt and a very large pan range; the real
  limits are set by the mount and the camera cable.
- Head control exists only through the robot server (`head_move`, `head_record`, `head_torque_limit`, raw ticks
  with the saved signs, clamped to the recorded limits). There is no "look at this point" yet.
- No check that the head moves correctly together with the arm teleop.

## About `xlerobot.urdf`

It includes the head as `head_pan_joint` (axis z, +/-183 deg) and `head_tilt_joint` (axis y, -43.5 to +83.1 deg)
under `top_base_link`, with the camera on `head_tilt_link`. For later hand-eye work: `head_camera_link` has x
pointing along the camera's view, which is right. But `head_camera_rgb_frame` / `head_camera_depth_frame`, with
rpy (0, -pi/2, -pi/2), come out with their z axis pointing sideways when the head is at rest (the optical axis
should be along `head_camera_link`'s x), so check that rotation before trusting those frames. In this URDF the
camera looks along -x of `top_base_link` at rest.

## Tests

`tests/felix_jr/test_base_follower.py` and `test_setup_head.py`: motor layout and IDs, the observation filter,
sign files, `configure`, the scan-based ID assignment (including "nothing new" and "two unknown servos"), the
nudge (direction, end of range, torque off after a failure) and the scripted setup run. They use a fake bus, so
nothing here has been run on the servos yet.
