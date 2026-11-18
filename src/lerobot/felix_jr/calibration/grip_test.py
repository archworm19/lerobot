"""How the gripper grips: closes on upright bottles of known size, the servo sampled, then a hold and a lift.

  uv run --extra felix_jr --extra felix_jr_sim --extra pyzmq-dep python -m lerobot.felix_jr.calibration.grip_test \
      tcp://<pi>:5555 --arm right --out-dir <scratch folder> [--model <calibration.urdf>] [--table-z Z]

The data for simulating the grip (`docs/sim2real_grasp.md`, "Next": the contacts, the servo's force, the fingers'
give): for each bottle, where the moving jaw stalls on it, the load and current the servo presses with, how they
change over a hold (the servo's overload protection), and whether it holds through a lift. The server needs protocol 6
(`grip`: the gripper's servo sampled on the motion thread).

Per bottle, in the terminal (the arm moves first, the bottle is placed after, so the arm never sweeps into it):

1. **The bottle:** a label, its diameter (calipers, mm), height (mm) and mass (g, the kitchen scale). Enter on a
   number keeps the last bottle's. Its depth in the jaws (the fingertips to its axis) is its radius and
   `DEPTH_MARGIN_M`, the bottle wholly between the fingers (`--depth-mm` sets one for every bottle instead).
2. **The grip pose:** from the side (the jaws closing across, horizontally, the wrist camera on top), the fingers as
   near level as reachable (`GRIP_TILTS_DEG`) `GRIP_AHEAD_M` ahead of the arm's base, the tips at `TIP_HEIGHT` of the bottle's height. The sim checks the way there and
   the pose (the gripper's lowest over the table, `MIN_CLEARANCE_M`); it asks before moving. The jaws open to the
   bottle's width plus `OPEN_MARGIN_M` (the gripper map).
3. **Stand the bottle** where its outline is drawn on the live wrist picture (`place`: its circle on the table, at
   the tips' height and at its top, from the model's grasp point for its width moved back by the depth); Enter in
   that window (`--no-window`: in the terminal, by eye).
4. **The closes:** to the map's contact for that width, then `PAST_CONTACT` units past it, then to 0, each sampled
   for `CLOSE_S` (position, load, current) with a wrist frame at the end; the stall (the last half second's median)
   printed as it goes.
5. **The hold:** at 0 for `HOLD_S`, sampled. The servo's overload protection can trip (it did at once on a 45 mm
   bottle, 2026-11-17): the server opens the jaw a little and says so (`"overload"`), and the trial records when.
6. **The lift:** up `LIFT_M` (the gripper level, the jaws as they are), wrist frames at once and after `LIFT_HOLD_S`;
   you say whether it held, slipped or dropped. Back down, the jaws open: take the bottle out.

`q` at any prompt ends (the arm goes to rest). Everything is saved as it goes in `--out-dir`: `results.json` (per
trial: the bottle, the poses, every close's samples and its stall, the hold, the lift) and `frames/`.
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lerobot.felix_jr.calibration import joint_views as jv
from lerobot.felix_jr.calibration.xle_robot_model import XleRobotModel, pinch_frame
from lerobot.felix_jr.server.client import FelixClient
from lerobot.felix_jr.server.protocol import DEFAULT_PORT

# the bottle's axis this far in from the tips, by default: its radius and this margin, the bottle wholly between the
# fingers (a 52 mm bottle at 15 mm, mostly beyond the tips, was pushed out as the claw closed: 2026-11-17)
DEPTH_MARGIN_M = 0.005
PREVIEW_PERIOD_S = 0.15  # the placement window's refresh
OUTLINE_COLORS = {"bottom": (255, 160, 0), "tips": (255, 64, 255), "top": (0, 255, 255)}  # RGB
GRIP_AHEAD_M = (0.25, 0.21, 0.17)  # the grip pose's pinch point ahead of the arm's base: the first reached
GRIP_TILTS_DEG = (0.0, 15.0, 30.0, 45.0)  # the fingers' tilt below level: the first reached (level, ideally)
TIP_HEIGHT = 0.5  # the tips at this fraction of the bottle's height ...
TIP_HEIGHT_M = (0.05, 0.12)  # ... within these heights over the table
OPEN_MARGIN_M = 0.015  # the jaws opened this much wider than the bottle
PAST_CONTACT = (0.0, 2.0, 5.0, 10.0)  # the closes: gripper.pos units past the map's contact, then 0
CLOSE_S = 2.0
HOLD_S = 30.0
LIFT_M = 0.04
LIFT_HOLD_S = 5.0
SAMPLE_S = 0.05
MIN_CLEARANCE_M = 0.02  # the gripper's lowest over the table, on the way and at the pose
MAX_VEL = 20.0
STALL_S = 0.5  # a close's stall: the median of its last this-many seconds
# the jaw this far (gripper.pos units) past the map's contact: the bottle is gone from between the fingers (a 52 mm
# bottle at 15 mm deep popped out at the close to 0, 2026-11-17; held, the jaw closed 9 units past contact at most)
SLIP_UNITS = 15.0


@dataclass
class Bottle:
    """What the person measured."""

    label: str
    diameter_mm: float
    height_mm: float
    mass_g: float
    depth_mm: float


def grip_pose(
    model: XleRobotModel,
    arm: str,
    rest: dict[str, float],
    table_z: float,
    tip_height_m: float,
    lift_m: float = 0.0,
    tilts_deg: tuple[float, ...] = GRIP_TILTS_DEG,
    aheads_m: tuple[float, ...] = GRIP_AHEAD_M,
) -> tuple[dict[str, float], float, float] | None:
    """The arm's joints for a side grip, with the fingers' tilt and the distance ahead it took; None if none reaches.

    The pinch point ahead of the arm's base (the first of `aheads_m` reached), `tip_height_m` + `lift_m` over the
    table; the fingers pointing ahead, tilted down by the first of `tilts_deg` reached (level is out of the wrist's
    range low and far: `wrist_flex` at its limit); the jaws closing across, horizontally (the hinge in the vertical
    plane of the fingers, the wrist camera on top). Each solve kept only on target (2 cm, 5 deg), within the joints'
    limits and off their bounds, the wrist camera clear of the arm.
    """
    base = model.frame_pose(f"{arm}_base_link", model.joint_vector())[:3, 3]
    forward, up = (
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )  # ahead is -x (`joint_views.low_poses`)
    face = model.wrist_face(arm) / np.linalg.norm(model.wrist_face(arm))
    rng = np.random.default_rng(0)
    for tilt in tilts_deg:
        t = np.radians(tilt)
        along = np.cos(t) * forward - np.sin(t) * up  # the fingers, toward the tips
        across = (
            np.sin(t) * forward + np.cos(t) * up
        )  # the hinge's side the wrist camera is on: up, square to them
        for ahead in aheads_m:
            target = np.array([base[0] - ahead, base[1], table_z + tip_height_m + lift_m])

            def residual(
                q: np.ndarray, target: np.ndarray = target, along=along, across=across
            ) -> np.ndarray:
                pinch = model.frame_pose(pinch_frame(arm), q)
                return np.r_[
                    (pinch[:3, 3] - target) / 0.005,
                    (pinch[:3, 2] - along) / 0.05,
                    (pinch[:3, :3] @ face - across) / 0.05,
                ]

            for k in range(8):
                start = np.array([0.0, 20.0, 20.0, -20.0, 90.0]) + (rng.normal(0.0, 25.0, 5) if k else 0.0)
                pose = jv._arm_solve(model, arm, rest, residual, start)
                q = model.joint_vector({arm: pose})
                pinch = model.frame_pose(pinch_frame(arm), q)
                off = np.degrees(np.arccos(np.clip(pinch[:3, 2] @ along, -1.0, 1.0)))
                turned = np.degrees(np.arccos(np.clip(pinch[:3, :3] @ face @ across, -1.0, 1.0)))
                if (
                    np.linalg.norm(pinch[:3, 3] - target) < 0.02
                    and off < 5.0
                    and turned < 5.0
                    and jv.within_limits(model, arm, pose)
                    and not jv.on_a_bound(model, arm, pose)
                    and jv.camera_clear_of_arm(model, arm, pose)
                ):
                    return pose, float(tilt), float(ahead)
    return None


def slipped(samples: list[dict[str, float]], contact: float) -> float | None:
    """When (s) the jaw went `SLIP_UNITS` past `contact`: the bottle slipped out; None if it did not."""
    return next((s["t"] for s in samples if s["pos"] < contact - SLIP_UNITS), None)


def overloaded(samples: list[dict[str, float]]) -> float | None:
    """When (s) the servo's overload protection tripped (`RealArms.grip` opened the jaw a little then); None if not."""
    return next((s["t"] for s in samples if s.get("overload")), None)


def stall(samples: list[dict[str, float]]) -> dict[str, float]:
    """A close's end: the median of its last `STALL_S` seconds (position, load, current)."""
    if not samples:
        return {}
    end = samples[-1]["t"]
    tail = [s for s in samples if s["t"] >= end - STALL_S] or samples[-1:]
    return {k: float(np.median([s[k] for s in tail])) for k in ("pos", "load_pct", "current_ma")}


class GripTest:
    """One arm's grip trials on the robot (the module docstring)."""

    def __init__(
        self,
        client: FelixClient,
        arm: str,
        out_dir: Path,
        model: XleRobotModel,
        table_z: float,
        ask=input,
        window: bool = True,
        depth_mm: float | None = None,
    ):
        """The model must have the arm's gripper map; checked against the robot.

        `window`: the placement shown on the wrist camera (`place`; else asked in the terminal). `depth_mm`: every
        bottle's axis this far in from the tips (None: its radius and `DEPTH_MARGIN_M`).
        """
        self.window, self.depth_mm = window, depth_mm
        self.client, self.arm, self.arm_id, self.out_dir, self.model = (
            client,
            arm,
            f"{arm}_arm",
            out_dir,
            model,
        )
        self.table_z, self.ask = table_z, ask
        if arm not in model.gripper_maps:
            raise ValueError(f"the model has no {arm} gripper map: a joint session first (README section 7)")
        model.check_robot(client)
        (out_dir / "frames").mkdir(parents=True, exist_ok=True)
        self.results_path = out_dir / "results.json"
        self.results: list[dict[str, Any]] = (
            json.loads(self.results_path.read_text()) if self.results_path.is_file() else []
        )
        self.rest = {k: float(v) for k, v in client.get_homes()[self.arm_id].items()}
        self.last: Bottle | None = None
        self.trial_index = len(self.results)  # the trial under way (its frames' prefix)

    def _save(self) -> None:
        self.results_path.write_text(json.dumps(self.results, indent=1))

    def _frame(self, name: str) -> str:
        path = f"frames/{self.trial_index:02d}_{name}.png"
        frame = self.client.wrist_capture(self.arm, color_format="png")
        cv2.imwrite(str(self.out_dir / path), cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR))
        return path

    def _number(self, prompt: str, last: float | None) -> float | None:
        while True:
            text = self.ask(f"{prompt}{f' [{last:g}]' if last is not None else ''}: ").strip()
            if text.lower() == "q":
                return None
            if not text and last is not None:
                return last
            try:
                return float(text)
            except ValueError:
                print("  a number, please (q stops)")

    def ask_bottle(self) -> Bottle | None:
        """The next bottle's measurements (None: stop)."""
        last = self.last
        label = self.ask(f"bottle label{f' [{last.label}]' if last else ''} (q stops): ").strip()
        if label.lower() == "q":
            return None
        values = []
        for prompt, field in (
            ("diameter mm", "diameter_mm"),
            ("height mm", "height_mm"),
            ("mass g", "mass_g"),
        ):
            value = self._number(f"  {prompt}", getattr(last, field) if last else None)
            if value is None:
                return None
            values.append(value)
        depth = self.depth_mm if self.depth_mm is not None else values[0] / 2 + 1000.0 * DEPTH_MARGIN_M
        return Bottle(label or (last.label if last else "bottle"), *values, depth)

    def bottle_outline(self, bottle: Bottle, q: np.ndarray) -> dict[str, np.ndarray]:
        """Where the bottle should stand, in the wrist picture at `q` (raw pixels; NaN where behind the camera).

        Its circle at the table, at the tips' height and at its top. Its axis vertical through the model's grasp point for its width (`XleRobotModel.grasp_offset`: midway between
        the tips closed on it), moved back along the fingers by its depth.
        """
        model, arm = self.model, self.arm
        pinch = model.frame_pose(pinch_frame(arm), q)
        center = (
            pinch
            @ np.r_[
                model.grasp_offset(arm, bottle.diameter_mm / 1000.0) - [0.0, 0.0, bottle.depth_mm / 1000.0],
                1.0,
            ]
        )
        r = bottle.diameter_mm / 2000.0
        angles = np.linspace(0.0, 2 * np.pi, 48)
        ring = np.c_[center[0] + r * np.cos(angles), center[1] + r * np.sin(angles)]
        heights = {"bottom": self.table_z, "tips": center[2], "top": self.table_z + bottle.height_mm / 1000.0}
        return {
            name: jv.wrist_pixels(model, arm, q, np.c_[ring, np.full(len(ring), z)])
            for name, z in heights.items()
        }

    def place(self, bottle: Bottle) -> bool:
        """The person stands the bottle where its outline is drawn on the live wrist picture; False: stopped.

        Enter (or space) in the window when it matches; `q` stops. Without `window`, asked in the terminal. The
        picture with the outline is kept (`frames/<trial>_placed.png`).
        """
        prompt = (
            f"  stand the bottle between the jaws, its axis {bottle.depth_mm:g} mm in from the tips (the moving jaw "
            "sweeps it onto the fixed one)"
        )
        if not self.window:
            return self.ask(prompt + "; Enter (q stops) ").strip().lower() != "q"
        print(prompt + ": line it up with the outline in the window, then Enter there (q stops)")
        state = self.client.get_state()[self.arm_id]
        q = self.model.joint_vector({self.arm: {k: float(v) for k, v in state.items()}})
        outline = self.bottle_outline(bottle, q)
        name = "grip_test: stand the bottle on the outline (Enter: placed, q: stop)"
        try:
            while True:
                picture = self.client.wrist_capture(self.arm, color_format="jpeg").color.copy()
                for part, uv in outline.items():
                    ok = np.isfinite(uv).all(axis=1)
                    if ok.sum() > 2:
                        cv2.polylines(
                            picture,
                            [np.round(uv[ok]).astype(np.int32)],
                            True,
                            OUTLINE_COLORS[part],
                            2,
                            cv2.LINE_AA,
                        )
                cv2.putText(
                    picture,
                    "orange: on the table  magenta: at the tips  cyan: its top",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(name, cv2.cvtColor(picture, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(int(1000 * PREVIEW_PERIOD_S)) & 0xFF
                if key in (13, 10, 32):
                    path = f"frames/{self.trial_index:02d}_placed.png"
                    cv2.imwrite(str(self.out_dir / path), cv2.cvtColor(picture, cv2.COLOR_RGB2BGR))
                    return True
                if key in (ord("q"), 27):
                    return False
        finally:
            cv2.destroyWindow(name)
            cv2.waitKey(1)

    def _move(self, pose: dict[str, float], why: str) -> bool:
        """Move the arm to `pose` (the gripper not touched) after the sim's check and the person's yes."""
        from .gripper_sweep import FingerRenderer, path_clearance

        start = {k: float(v) for k, v in self.client.get_state()[self.arm_id].items()}
        fingers = FingerRenderer(self.model, self.arm)
        try:
            at_start, on_the_way = path_clearance(
                fingers, start, {**pose, "gripper.pos": start["gripper.pos"]}, self.table_z
            )
        finally:
            fingers.close()
        print(f"  {why}: the gripper's lowest over the table {100 * on_the_way:.1f} cm on the way")
        if on_the_way < min(MIN_CLEARANCE_M, at_start):
            print(f"  refused: under {100 * MIN_CLEARANCE_M:.0f} cm (--table-z?)")
            return False
        if self.ask(f"  move the arm ({why})? [y/N] ").strip().lower() != "y":
            return False
        status = self.client.move_joints({self.arm_id: pose}, max_vel=MAX_VEL)
        return status["state"] == "done"

    def trial(self, bottle: Bottle) -> bool:
        """One bottle's closes, hold and lift (the module docstring); False: stopped."""
        model, arm = self.model, self.arm
        self.trial_index = len(self.results)
        tip = float(np.clip(TIP_HEIGHT * bottle.height_mm / 1000.0, *TIP_HEIGHT_M))
        found = grip_pose(model, arm, self.rest, self.table_z, tip)
        lifted = (
            None
            if found is None
            else grip_pose(model, arm, self.rest, self.table_z, tip, LIFT_M, (found[1],), (found[2],))
        )
        if found is None or lifted is None:
            print("  no side grip pose reaches there: a taller bottle (the tips higher), or --table-z")
            return True
        pose, tilt, ahead = found
        lift = lifted[0]
        print(
            f"  the grip pose: {100 * ahead:.0f} cm ahead of the arm's base, the fingers {tilt:g} deg below level"
        )
        width = bottle.diameter_mm / 1000.0
        open_pos = model.gripper_pos_for(arm, model.opening_for_width(arm, width + OPEN_MARGIN_M))
        contact = model.gripper_pos_for(arm, model.opening_for_width(arm, width))
        if not self._move(pose, f"to the grip pose, the tips {100 * tip:.0f} cm over the table"):
            return False
        self.client.grip(self.arm_id, open_pos, seconds=1.5)
        record: dict[str, Any] = {
            "bottle": asdict(bottle),
            "time": time.time(),
            "tip_height_m": tip,
            "pose": pose,
            "tilt_deg": tilt,
            "ahead_m": ahead,
            "open_pos": open_pos,
            "contact_pos": contact,
            "closes": [],
        }
        if not self.place(bottle):
            return False
        record["frame_open"] = self._frame("open")
        self.results.append(record)  # (saved as it fills)
        for past in PAST_CONTACT:
            target = max(0.0, contact - past)
            samples = self.client.grip(self.arm_id, target, seconds=CLOSE_S, period=SAMPLE_S)
            end = stall(samples)
            record["closes"].append(
                {
                    "target": target,
                    "past_contact": past,
                    "samples": samples,
                    "stall": end,
                    "frame": self._frame(f"close{past:g}"),
                }
            )
            print(
                f"  close to {target:5.1f} ({past:g} past contact {contact:.1f}): stalled at {end['pos']:5.1f}, "
                f"load {end['load_pct']:+.1f} %, {end['current_ma']:.0f} mA"
            )
            self._save()
            if (t := overloaded(samples)) is not None:
                record["overload"] = {"during": f"close {past:g} past contact", "t": t}
                break
            if (t := slipped(samples, contact)) is not None:
                record["slipped"] = {"during": f"close {past:g} past contact", "t": t}
                break
        if "slipped" not in record and "overload" not in record:
            samples = self.client.grip(self.arm_id, 0.0, seconds=HOLD_S, period=0.2)
            loads = [s["load_pct"] for s in samples]
            record["hold"] = {"samples": samples, "first": stall(samples[:5]), "last": stall(samples)}
            print(
                f"  hold {HOLD_S:g} s at 0: load {np.median(loads[:5]):+.1f} -> {np.median(loads[-5:]):+.1f} %"
            )
            if (t := overloaded(samples)) is not None:
                record["overload"] = {"during": "the hold at 0", "t": t}
            elif (t := slipped(samples, contact)) is not None:
                record["slipped"] = {"during": "the hold at 0", "t": t}
            self._save()
        if "overload" in record:
            print(
                f"  OVERLOAD: the servo's protection tripped during {record['overload']['during']} "
                f"({record['overload']['t']:.1f} s in; the jaw opened a little): no lift"
            )
            self._save()
        elif "slipped" in record:
            print(
                f"  SLIPPED OUT during {record['slipped']['during']} ({record['slipped']['t']:.1f} s in): no lift"
            )
            self._save()
        elif self._move(lift, f"the lift, {100 * LIFT_M:.0f} cm up"):
            record["lift"] = {"frame_up": self._frame("lift0")}
            time.sleep(LIFT_HOLD_S)
            record["lift"]["frame_after"] = self._frame(f"lift{LIFT_HOLD_S:g}")
            held = ""
            while held not in ("held", "slipped", "dropped"):
                held = self.ask("  did it hold? [held/slipped/dropped]: ").strip().lower()
            record["lift"]["result"] = held
            self._save()
            self._move(pose, "back down")
        self.client.grip(self.arm_id, open_pos, seconds=1.5)
        self.ask("  take the bottle out; Enter ")
        self.last = bottle
        self._save()
        return True

    def summary(self) -> str:
        """Per trial: the bottle, the stalls past contact, the hold's load, the lift."""
        lines = [
            "| bottle | mm | g | contact | stalls (pos / load %) | hold load % | lift |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in self.results:
            b = r["bottle"]
            stalls = ", ".join(f"{c['stall']['pos']:.1f}/{c['stall']['load_pct']:+.0f}" for c in r["closes"])
            hold = r.get("hold", {})
            held = f"{hold['first']['load_pct']:+.0f} -> {hold['last']['load_pct']:+.0f}" if hold else ""
            lines.append(
                f"| {b['label']} | {b['diameter_mm']:g} | {b['mass_g']:g} | {r['contact_pos']:.1f} | {stalls} | {held} | "
                f"{
                    r.get('lift', {}).get('result', '')
                    or ('slipped: ' + r['slipped']['during'] if 'slipped' in r else '')
                    or ('overload: ' + r['overload']['during'] if 'overload' in r else '')
                } |"
            )
        return "\n".join(lines)


def main() -> None:
    """Command line entry point; see the module docstring."""
    from lerobot.felix_jr.harness import api

    from .gripper_sweep import table_height

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("endpoint", nargs="?", default=f"tcp://localhost:{DEFAULT_PORT}")
    parser.add_argument("--arm", choices=["left", "right"], required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path, default=api.MODEL, help="the calibration (default: the harness's)"
    )
    parser.add_argument(
        "--table-z", type=float, default=None, help="the table's height in the root frame (m)"
    )
    parser.add_argument(
        "--depth-mm",
        type=float,
        default=None,
        help="every bottle's axis this far in from the tips (default: its radius + 5)",
    )
    parser.add_argument("--no-window", action="store_true", help="place the bottles by eye, no wrist picture")
    args = parser.parse_args()
    model = XleRobotModel.load(args.model)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with FelixClient(args.endpoint, timeout_s=20.0) as client, client.session():
        test = GripTest(
            client,
            args.arm,
            args.out_dir,
            model,
            table_height(model, args.arm, args.table_z),
            window=not args.no_window,
            depth_mm=args.depth_mm,
        )
        print(f"{args.arm} arm: the table at {100 * test.table_z:.1f} cm; {len(test.results)} trials so far")
        try:
            while (bottle := test.ask_bottle()) is not None and test.trial(bottle):
                pass
        finally:
            client.home(max_vel=MAX_VEL)
    print(test.summary())
    print(f"saved to {args.out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
