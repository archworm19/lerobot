"""Give an arm's parts their shapes in the calibration, from measurements: the gripper body and the forearm as boxes.

  uv run python -m lerobot.felix_jr.calibration.arm_shapes tcp://<pi>:5555 --start-from <calibration.urdf>
      --out-dir <folder> [--arm right] [--parts gripper_body,forearm] [--max-vel 30] [--floor-z Z]

The grasping's line-of-sight checks (`docs/object_grasping.md`) need the gripper's and the forearm's shape. Each is a
box square to its link (`ArmBox`): measure it with a ruler, type the sizes, and check it drawn over the head camera's
picture. (Fitting the boxes to clicked outlines proved fragile, 2026-11-06: the forearm's soft sleeve, and clicks on
edges inside the outline that flattened the boxes.)

Without `--arm`, the terminal asks which arm. Then, for each part in turn (the gripper body, then the forearm):

1. The head turns to the part (the arm stays where it is), and a frame is taken.
2. The terminal asks for its sizes in cm (Enter keeps the ones shown: the calibration's box, if it has one):
     gripper body  length (from the wrist toward the tips), width (across the jaws, the way they close), depth (along
                   the jaws' hinge); placed from the wrist roll's axis toward the tips, centered on the link
     forearm       length (from the elbow toward the wrist), width (side to side, along the elbow's axis), height
                   (from the joints' line down); centered between the elbow and the wrist and side to side, its top on
                   the line through the elbow's and the wrist's axes
3. The box is drawn orange over the frame (in a window, and as `<arm>_<part>.png` in `--out-dir`), with an arrow from
   its center along each direction, and the terminal asks:
     y            it sits on the part: keep it, go on to the next part
     s L W D      new sizes (cm), in the order asked
     m L W D      shift it (cm) along the arrows drawn (negative: against them)
     v            another view: the arm and the head move to the next planned view (`LiveView`)
     h N E        turn the head (deg: the neck, + left; the eyes, + down)
     q            stop: nothing is written

The output, `--out-dir/calibration.urdf`, is `--start-from` with the parts' boxes (`XleRobotModel.with_boxes`, in
`<felix_calibration>`), and `arm_shapes.md` lists them; for the other arm, run again with `--start-from` that file.
The fingers are not measured here: the grasping puts stand-ins where the calibration puts the tips
(`harness.grasping.finger_boxes`). The script runs inside a client session, so on Ctrl+C or a crash the robot goes
to rest.
"""

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from lerobot.felix_jr.calibration.live_view import INTRINSICS_KEYS, LiveView, set_floor
from lerobot.felix_jr.calibration.merge_urdf import ARMS
from lerobot.felix_jr.calibration.xle_robot_model import HEAD_JOINTS, ArmBox, XleRobotModel
from lerobot.felix_jr.harness.look import aim_head
from lerobot.felix_jr.server.client import FelixClient
from lerobot.felix_jr.server.protocol import DEFAULT_PORT

CALIBRATION_FILE = "calibration.urdf"
REPORT_FILE = "arm_shapes.md"
BOX_COLOR = (255, 140, 0)  # RGB: orange
ARROW_COLORS = ((255, 80, 80), (80, 255, 80), (80, 160, 255))  # RGB: the three directions, in the order asked
ARROW_M = 0.04  # the arrows drawn from a box's center
AIM_MARGIN_PX = 60  # a part whose center is not this far inside the picture: the head turns to it
START_SIZE_M = 0.04  # each way, for aiming the head at a part that has no box yet


@dataclass(frozen=True)
class Part:
    """How a part's box is measured: along which of its link's axes each size runs, and where it sits by default.

    `axes[i]` is the link axis (0 x, 1 y, 2 z) of the `i`-th measure (`names[i]`, `help[i]`) and `signs[i]` the way
    along it its arrow (and a positive shift) points.
    """

    link: str  # without the arm: "gripper_link"
    names: tuple[str, str, str]
    help: tuple[str, str, str]
    axes: tuple[int, int, int]
    signs: tuple[float, float, float]


PARTS = {
    # the gripper link: z from the tips to the wrist roll's axis (the pinch point is 10 cm along -z), x across the
    # jaws (the moving jaw's side +x), y along the hinge
    "gripper_body": Part(
        "gripper_link",
        ("length", "width", "depth"),
        ("from the wrist toward the tips", "across the jaws, the way they close", "along the jaws' hinge"),
        (2, 0, 1),
        (-1.0, 1.0, 1.0),
    ),
    # the lower arm link: x from the wrist (13.4 cm along -x) to the elbow, z along the flex joints' axes, y up (both
    # arms, in every pose with the forearm above the wrist: the forearm hangs below the joints' line, along -y)
    "forearm": Part(
        "lower_arm_link",
        ("length", "width", "height"),
        (
            "from the elbow toward the wrist",
            "side to side, along the elbow's axis",
            "from the elbow-wrist joint line down",
        ),
        (0, 2, 1),
        (-1.0, 1.0, -1.0),
    ),
}


def _point(p: np.ndarray) -> tuple[float, float, float]:
    return float(p[0]), float(p[1]), float(p[2])


def part_link(arm: str, part: str) -> str:
    """The link `arm`'s `part` is fixed to."""
    return f"{arm}_{PARTS[part].link}"


def _origin_in(model: XleRobotModel, parent: str, child: str) -> np.ndarray:
    """Where `child`'s origin is in `parent`'s frame, at the model's zero pose."""
    q = model.joint_vector()
    parent_frame, child_frame = model.frame_pose(parent, q), model.frame_pose(child, q)
    return parent_frame[:3, :3].T @ (child_frame[:3, 3] - parent_frame[:3, 3])


def base_center(model: XleRobotModel, arm: str, part: str, sizes: Sequence[float]) -> np.ndarray:
    """Where the box of `sizes` (m, in the order asked) sits unshifted, in its link's frame.

    The gripper body from the wrist roll's axis (the link's origin) toward the tips, centered across. The forearm
    centered between the elbow (its link's origin) and the wrist, and side to side, its top on the joints' line: it
    hangs below the line through the flex joints' axes (bowl3, 2026-11-07: centered on it, it sat on top of the arm).
    """
    spec, center = PARTS[part], np.zeros(3)
    if part == "gripper_body":
        center[spec.axes[0]] = spec.signs[0] * sizes[0] / 2
    else:
        center = 0.5 * _origin_in(model, part_link(arm, part), f"{arm}_wrist_link")
        center[spec.axes[1]] = 0.0
        center[spec.axes[2]] = spec.signs[2] * sizes[2] / 2
    return center


def measured_box(
    model: XleRobotModel,
    arm: str,
    part: str,
    sizes: Sequence[float],
    shifts: Sequence[float] = (0.0, 0.0, 0.0),
) -> ArmBox:
    """`arm`'s `part` as a box of `sizes` (m, in the order asked: `PARTS[part].names`), shifted by `shifts` (m)."""
    spec = PARTS[part]
    center = base_center(model, arm, part, sizes)
    size = np.zeros(3)
    for i, axis in enumerate(spec.axes):
        size[axis] = sizes[i]
        center[axis] += spec.signs[i] * shifts[i]
    return ArmBox(part_link(arm, part), _point(center), _point(size))


def measures_of(model: XleRobotModel, arm: str, part: str, box: ArmBox) -> tuple[list[float], list[float]]:
    """The sizes and shifts (m, in the order asked) that give `box` (`measured_box` the other way)."""
    spec = PARTS[part]
    sizes = [float(box.size[axis]) for axis in spec.axes]
    center = base_center(model, arm, part, sizes)
    shifts = [float(spec.signs[i] * (box.center[axis] - center[axis])) for i, axis in enumerate(spec.axes)]
    return sizes, shifts


def parse_numbers(text: str, n: int = 3) -> list[float] | None:
    """`n` numbers from `text` (spaces or commas between them); None if it is not that."""
    try:
        values = [float(v) for v in text.replace(",", " ").split()]
    except ValueError:
        return None
    return values if len(values) == n else None


def report(model: XleRobotModel, arm: str, boxes: dict[str, ArmBox]) -> list[str]:
    """`arm_shapes.md`: each part's box, as measured (the sizes in the order asked, the shifts) and in its link."""
    lines = [
        f"# Arm shapes: {arm} arm",
        "",
        "| part | sizes (cm) | shifted (cm) | size x y z in the link (mm) | center (mm) | link |",
        "|---|---|---|---|---|---|",
    ]
    for part, box in boxes.items():
        sizes, shifts = measures_of(model, arm, part, box)
        names = PARTS[part].names
        lines.append(
            f"| {part} | {', '.join(f'{n} {v * 100:.1f}' for n, v in zip(names, sizes, strict=True))} | "
            f"{', '.join(f'{round(v * 100, 1) + 0.0:+.1f}' for v in shifts)} | {' x '.join(f'{v * 1000:.0f}' for v in box.size)} | "
            f"({', '.join(f'{v * 1000:.0f}' for v in box.center)}) | `{box.parent}` |"
        )
    return lines


def box_outline(
    model: XleRobotModel, q: np.ndarray, root_from_box: np.ndarray, half: np.ndarray
) -> np.ndarray | None:
    """A box's outline in the image at `q`, as a closed polygon of pixels; None if a corner is behind the camera.

    The convex hull of its projected corners: a convex box's silhouette exactly.
    """
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    corners = (signs * half) @ root_from_box[:3, :3].T + root_from_box[:3, 3]
    uv, z = model.project(corners, q)
    if not (z > 0).all() or not np.isfinite(uv).all():
        return None
    return cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2).astype(float)


def shape_name(arm: str, part: str, shown_arm: str) -> str:
    """A shape's label on the picture: its part ("gripper body"), with the arm when it is not the one shown."""
    name = part.replace("_", " ")
    return name if arm == shown_arm else f"{arm} {name}"


def draw_label(
    out: np.ndarray,
    polygon: np.ndarray,
    text: str,
    color: tuple[int, int, int],
    taken: list[tuple[int, int, int, int]] | None = None,
) -> None:
    """Write `text` beside a shape's outline, in its color, on `out` (RGB, in place).

    It goes above the shape, else below, right or left of it: the first place clear of the labels in `taken`
    (boxes x0, y0, x1, y1, added to as labels are drawn), so the names of shapes close together do not overlap.
    """
    if not text or not np.isfinite(polygon).all():
        return
    (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    left, top = polygon.min(axis=0)
    right, bottom = polygon.max(axis=0)
    middle_x, middle_y = (left + right) / 2, (top + bottom) / 2
    places = [  # the text's bottom-left corner
        (middle_x - width / 2, top - 6),
        (middle_x - width / 2, bottom + height + 6),
        (right + 6, middle_y + height / 2),
        (left - width - 6, middle_y + height / 2),
    ]
    taken = [] if taken is None else taken
    boxes = [(int(x), int(y) - height - 2, int(x) + width, int(y) + 3) for x, y in places]
    clear = [box for box in boxes if not any(_overlap(box, other) for other in taken)]
    x0, y0, x1, y1 = clear[0] if clear else boxes[0]
    taken.append((x0, y0, x1, y1))
    org = (x0, y1 - 3)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def annotate(
    image: np.ndarray,
    lines: list[str],
    shapes: Sequence[tuple[np.ndarray, tuple[int, int, int], str]],
    clicks: Sequence[tuple[tuple[float, float], str]],
) -> np.ndarray:
    """Copy of the RGB `image` with the fitted shapes, this frame's clicks and text.

    The shapes are closed polygons, each in its color with its name above it; the clicks green crosses, labelled.
    """
    out = image.copy()
    taken: list[tuple[int, int, int, int]] = []
    for polygon, color, name in shapes:
        if np.isfinite(polygon).all():
            points = np.round(polygon).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [points], True, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.polylines(out, [points], True, color, 1, cv2.LINE_AA)
            draw_label(out, polygon, name, color, taken)
    for (u, v), label in clicks:
        centre = (int(round(u)), int(round(v)))
        cv2.drawMarker(out, centre, (0, 255, 0), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
        cv2.putText(
            out,
            label,
            (centre[0] + 6, centre[1] + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    for row, text in enumerate(lines):
        org = (8, 20 + 20 * row)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ---- the session --------------------------------------------------------------------------------------------

COMMANDS = "y keep | s L W D sizes (cm) | m L W D shift (cm) | v another view | h N E head (deg) | q quit"


class ArmShapesSession(LiveView):
    """Measure an arm's parts as boxes and check them over the head camera's frame. See the module docstring.

    `ask` puts a question to the person and returns the answer (`input`); `show` displays an RGB image.
    """

    no_more_views = "no more planned views: turn the head (h) instead"

    def __init__(
        self,
        client: FelixClient,
        start_from: Path,
        arm: str,
        out_dir: Path,
        show: Callable[[np.ndarray], None],
        ask: Callable[[str], str] = input,
        parts: Sequence[str] = tuple(PARTS),
        n_views: int = 12,
        max_vel: float = 30.0,
        floor_z: float | None = None,
        color_format: str = "png",
    ):
        """Load the calibration and check it is for this robot (`RobotMismatchError` before anything moves)."""
        unknown = set(parts) - set(PARTS)
        if unknown or not parts:
            raise ValueError(f"parts must be some of {tuple(PARTS)}, not {sorted(unknown) or 'none'}")
        self.parts = [p for p in PARTS if p in parts]
        self.start_from, self.out_dir, self.ask = start_from, out_dir, ask
        self.calibration = XleRobotModel.load(start_from)  # what is written out, with the boxes
        self.calibration.check_robot(client)
        model = XleRobotModel.load(start_from)  # the one the views are planned with
        self.notes: list[str] = []
        capture = client.capture(color_format=color_format)
        intrinsics = {k: capture.calibration["color_intrinsics"][k] for k in INTRINSICS_KEYS}
        if model.intrinsics != intrinsics:
            self.notes.append("the camera's intrinsics differ from the calibration's: using the camera's")
            model.intrinsics = self.calibration.intrinsics = intrinsics
        set_floor(model, arm, floor_z, self.notes.append)
        super().__init__(client, model, arm, show, capture, n_views, max_vel, color_format=color_format)
        self.boxes: dict[str, ArmBox] = dict(self.calibration.boxes.get(arm, {}))
        self.part: str | None = None  # the part being measured

    # ---- showing --------------------------------------------------------------------------------------------

    def render(self) -> None:
        """Show the frame with the part's box, its arrows (labelled with the measures' names) and the commands."""
        lines = [COMMANDS]
        if self.message:
            lines.append(self.message)
        shapes: list[tuple[np.ndarray, tuple[int, int, int], str]] = []
        image = self.capture.color
        box = self.boxes.get(self.part) if self.part else None
        if box is not None and self.part is not None:
            q = self.current_q()
            frame = self.model.box_frame(box, self.arm, q)
            polygon = box_outline(self.model, q, frame, 0.5 * np.asarray(box.size))
            if polygon is not None:
                shapes.append((polygon, BOX_COLOR, shape_name(self.arm, self.part, self.arm)))
            sizes, _ = measures_of(self.calibration, self.arm, self.part, box)
            spec = PARTS[self.part]
            lines.insert(
                1,
                f"{self.part.replace('_', ' ')}: "
                + " x ".join(f"{n} {v * 100:.1f}" for n, v in zip(spec.names, sizes, strict=True))
                + " cm",
            )
        image = annotate(image, lines, shapes, [])
        if box is not None and self.part is not None:
            self._draw_arrows(image, box, PARTS[self.part])
        if self.part is not None:
            cv2.imwrite(
                str(self.out_dir / f"{self.arm}_{self.part}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            )
        self.show(image)

    def _draw_arrows(self, image: np.ndarray, box: ArmBox, spec: Part) -> None:
        """An arrow from the box's center along each measure's direction, in its color, named (in place)."""
        q = self.current_q()
        frame = self.model.box_frame(box, self.arm, q)
        ends = [frame[:3, 3]]
        for i, axis in enumerate(spec.axes):
            ends.append(frame[:3, 3] + frame[:3, axis] * spec.signs[i] * ARROW_M)
        uv, z = self.model.project(np.array(ends), q)
        if not (z > 0).all():
            return
        start = tuple(int(round(v)) for v in uv[0])
        for i, color in enumerate(ARROW_COLORS):
            end = tuple(int(round(v)) for v in uv[i + 1])
            cv2.arrowedLine(image, start, end, (0, 0, 0), 4, cv2.LINE_AA, tipLength=0.2)
            cv2.arrowedLine(image, start, end, color, 2, cv2.LINE_AA, tipLength=0.2)
            cv2.putText(
                image, spec.names[i], (end[0] + 4, end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3
            )
            cv2.putText(image, spec.names[i], (end[0] + 4, end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # ---- moving ---------------------------------------------------------------------------------------------

    def aim_at(self, part: str) -> None:
        """Turn the head to the part's box (or where one would start), if its center is not well inside the picture."""
        box = self.boxes.get(part) or measured_box(self.calibration, self.arm, part, [START_SIZE_M] * 3)
        q = self.current_q()
        center = self.model.box_frame(box, self.arm, q)[:3, 3]
        uv, z = self.model.project(center, q)
        if z[0] > 0 and self.model.in_view(uv[0], margin=AIM_MARGIN_PX):
            self.render()
            return
        head = {n: int(self.capture.state["head"][n]["ticks"]) for n in HEAD_JOINTS}
        ticks, miss_deg = aim_head(self.model, head, center)
        self.message = f"turned the head to the {part.replace('_', ' ')}" + (
            f" (it points {miss_deg:.0f} deg off: h to turn it, v for another view)" if miss_deg > 5.0 else ""
        )
        self.refresh(self._move_head_to(ticks))

    # ---- measuring ------------------------------------------------------------------------------------------

    def ask_sizes(self, part: str) -> list[float] | None:
        """The part's sizes (m) from the person; Enter keeps its box's. None on q."""
        spec = PARTS[part]
        current = (
            measures_of(self.calibration, self.arm, part, self.boxes[part])[0] if part in self.boxes else None
        )
        names = ", ".join(f"{n} ({h})" for n, h in zip(spec.names, spec.help, strict=True))
        while True:
            shown = f" [{' '.join(f'{v * 100:.1f}' for v in current)}]" if current else ""
            answer = self.ask(f"{self.arm} {part.replace('_', ' ')}, in cm: {names}{shown}: ").strip()
            if answer == "q":
                return None
            if not answer and current:
                return current
            values = parse_numbers(answer)
            if values is not None and all(v > 0 for v in values):
                return [v / 100 for v in values]
            print("three sizes in cm, e.g. 7 3.5 3 (or q to stop)")

    def measure(self, part: str) -> bool:
        """Measure one part until the person keeps it. False on q."""
        self.part = part
        self.message = ""
        self.aim_at(part)
        sizes = self.ask_sizes(part)
        if sizes is None:
            return False
        shifts = (
            measures_of(self.calibration, self.arm, part, self.boxes[part])[1]
            if part in self.boxes
            else [0.0] * 3
        )
        while True:
            self.boxes[part] = measured_box(self.calibration, self.arm, part, sizes, shifts)
            self.render()
            answer = self.ask(
                f"{part.replace('_', ' ')} drawn: does it sit on the part? {COMMANDS}: "
            ).strip()
            command, _, rest = answer.partition(" ")
            values = parse_numbers(rest, 2 if command == "h" else 3)
            if command == "y":
                return True
            if command == "q":
                return False
            if command == "s" and values is not None and all(v > 0 for v in values):
                sizes = [v / 100 for v in values]
            elif command == "m" and values is not None:
                shifts = [s + v / 100 for s, v in zip(shifts, values, strict=True)]
            elif command == "v":
                self.next_pose()
            elif command == "h" and values is not None:
                self.jog(neck_deg=values[0], eyes_deg=values[1])
            else:
                print(f"not understood: {answer!r}")

    def run(self) -> Path | None:
        """Measure every part, then write the calibration and the report. None if the person stopped (q)."""
        for part in self.parts:
            if not self.measure(part):
                return None
        model = self.calibration.with_boxes(self.arm, {p: self.boxes[p] for p in self.parts})
        path = self.out_dir / CALIBRATION_FILE
        model.save(path)
        (self.out_dir / REPORT_FILE).write_text(
            "\n".join(report(model, self.arm, model.boxes[self.arm])) + "\n"
        )
        return path


def main() -> None:
    """Command line entry point; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("endpoint", nargs="?", default=f"tcp://localhost:{DEFAULT_PORT}")
    parser.add_argument("--arm", choices=ARMS, help="the arm (asked if not given)")
    parser.add_argument(
        "--start-from", type=Path, required=True, help="the calibration.urdf to add the boxes to"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--parts", default=",".join(PARTS), help=f"what to measure, comma-separated, of {','.join(PARTS)}"
    )
    parser.add_argument("--max-vel", type=float, default=30.0, help="arm speed limit, deg/s (30)")
    parser.add_argument("--floor-z", type=float, help="floor height in the robot's root frame, meters")
    args = parser.parse_args()
    if (args.out_dir / CALIBRATION_FILE).resolve() == args.start_from.resolve():
        parser.error("--out-dir would overwrite --start-from: pick another folder")
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    if not parts or set(parts) - set(PARTS):
        parser.error(f"--parts must be some of {','.join(PARTS)}")
    arm = args.arm
    while arm not in ARMS:
        arm = input(f"which arm? ({' or '.join(ARMS)}): ").strip()

    window = "arm shapes (answer in the terminal)"

    def show(image: np.ndarray) -> None:
        cv2.imshow(window, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(50)  # draw it before the terminal waits

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with FelixClient(args.endpoint, timeout_s=20.0) as client, client.session():
        session = ArmShapesSession(
            client,
            args.start_from,
            arm,
            args.out_dir,
            show,
            parts=parts,
            max_vel=args.max_vel,
            floor_z=args.floor_z,
        )
        for note in session.notes:
            print(note)
        try:
            path = session.run()
        finally:
            cv2.destroyAllWindows()
    if path is None:
        print("stopped: nothing written")
    else:
        print(f"Wrote {path} and {args.out_dir / REPORT_FILE}")
        print((args.out_dir / REPORT_FILE).read_text())


if __name__ == "__main__":
    main()
