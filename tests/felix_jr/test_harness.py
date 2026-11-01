"""The harness on a synthetic table scene: a camera 1 m above the root origin, pitched 50 degrees down, looking
along +x at a table (z = 0.3 m) with a box on it (top at z = 0.4 m, x 0.6..0.75 m, y -0.1..0.1 m)."""

import json
import pickle

import cv2
import numpy as np
import pytest

from lerobot.felix_jr.harness import (
    BLOCKED,
    FREE,
    UNKNOWN,
    Capture,
    Grid,
    Plan,
    Points,
    Poses,
    Reach,
    Scene,
    TableEstimate,
    View,
    Workspace,
    check_path,
    contact_sheet,
    focus,
    full_view,
    locate_grid,
    locate_pixels,
    make_grid,
    render,
    scan as scan_module,
    table_height,
    zoom,
)

H, W = 120, 160
FX = FY = 100.0
PPX, PPY = 80.0, 60.0
CAMERA = np.array([0.0, 0.0, 1.0])
TABLE_Z, BOX_Z = 0.3, 0.4
BOX_X, BOX_Y = (0.6, 0.75), (-0.1, 0.1)


def camera_pose(pitch_deg: float = 50.0) -> np.ndarray:
    a = np.radians(pitch_deg)
    z = np.array([np.cos(a), 0.0, -np.sin(a)])  # optical axis: forward and down
    x = np.array([0.0, -1.0, 0.0])  # image right = root -y
    pose = np.eye(4)
    pose[:3, :3] = np.stack([x, np.cross(z, x), z], axis=1)
    pose[:3, 3] = CAMERA
    return pose


def make_capture(box: bool = True) -> Capture:
    pose = camera_pose()
    vv, uu = np.mgrid[0:H, 0:W]
    d = np.stack([(uu - PPX) / FX, (vv - PPY) / FY, np.ones((H, W))], axis=-1) @ pose[:3, :3].T
    with np.errstate(divide="ignore", invalid="ignore"):
        t_table = (TABLE_Z - CAMERA[2]) / d[..., 2]
        t_box = (BOX_Z - CAMERA[2]) / d[..., 2]
    depth = np.where(t_table > 0, t_table, np.nan)  # optical z of the hit = t, since d_camera has z = 1
    if box:
        hit = CAMERA + t_box[..., None] * d
        on_box = (hit[..., 0] >= BOX_X[0]) & (hit[..., 0] <= BOX_X[1]) & (np.abs(hit[..., 1]) <= BOX_Y[1])
        depth = np.where(on_box & (t_box > 0), t_box, depth)
    color = np.full((H, W, 3), 90, dtype=np.uint8)
    return Capture(Scene(color, depth.astype(np.float32), FX, FY, PPX, PPY), pose, {}, {}, 1.0)


@pytest.fixture
def capture() -> Capture:
    return make_capture()


def box_pixels(capture: Capture) -> np.ndarray:
    """(H, W) bool: pixels showing the box top."""
    with np.errstate(invalid="ignore"):
        return np.abs(capture.scene.depth_m - make_capture(box=False).scene.depth_m) > 1e-4


class TestCapture:
    def test_frames_round_trip(self, capture):
        p = np.array([[0.7, 0.05, 0.3], [1.0, -0.2, 0.5]])
        np.testing.assert_allclose(capture.to_root(capture.to_camera(p)), p, atol=1e-12)

    def test_project_matches_depth(self, capture):
        """A table point projects to a pixel whose depth unprojects back to it."""
        p = np.array([0.9, 0.2, TABLE_Z])
        uv = capture.project(p)
        back = capture.to_root(capture.scene.pixel_to_xyz(uv))
        np.testing.assert_allclose(back, p, atol=0.01)

    def test_rays_hit_the_projected_point(self, capture):
        p = np.array([0.8, -0.1, TABLE_Z])
        origin, d = capture.rays(capture.project(p))
        np.testing.assert_allclose(np.cross(d, p - origin), 0, atol=1e-9)

    def test_save_and_load(self, capture, tmp_path):
        capture.save(tmp_path)
        loaded = Capture.load(tmp_path)
        np.testing.assert_array_equal(loaded.scene.color, capture.scene.color)
        np.testing.assert_allclose(loaded.scene.depth_m, capture.scene.depth_m, atol=6e-4, equal_nan=True)
        np.testing.assert_allclose(loaded.x_root_camera, capture.x_root_camera)
        assert loaded.timestamp == capture.timestamp


class TestView:
    def test_zoom_maps_pixels_both_ways(self, capture):
        view = zoom(full_view(capture), (40, 30, 80, 60))
        assert (view.u0, view.v0, view.width, view.height) == (40, 30, 80, 60)
        assert view.size == (W, H)  # fits the parent's size: scale 2
        uv = np.array([[50.0, 35.0], [100.0, 80.0]])
        np.testing.assert_allclose(view.to_capture(view.to_view(uv)), uv)
        np.testing.assert_allclose(view.to_view(np.array([40.0, 30.0])), [0, 0])

    def test_zoom_of_a_zoom_and_clipping(self, capture):
        inner = zoom(
            zoom(full_view(capture), (40, 30, 80, 60)), (0, 0, 80, 60)
        )  # top-left quarter of the zoom
        assert (inner.u0, inner.v0, inner.width, inner.height) == (40, 30, 40, 30)
        clipped = zoom(full_view(capture), (-50, -50, 100, 100))
        assert (clipped.u0, clipped.v0) == (0, 0) and clipped.width == 50 and clipped.height == 50


class TestTableHeight:
    def test_finds_the_table_not_the_box(self, capture):
        est = table_height(capture)
        assert est.h == pytest.approx(TABLE_Z, abs=0.002)
        assert est.tilt_deg < 0.5 and est.spread_m < 0.005
        assert not (est.inlier_mask & box_pixels(capture)).any()
        assert est.n_inliers == int(est.inlier_mask.sum()) > 1000

    def test_min_h_skips_the_table(self, capture):
        assert table_height(capture, min_h=0.35).h == pytest.approx(BOX_Z, abs=0.002)

    def test_tilt_shows_a_wrong_camera_pose(self, capture):
        """The same depth with a pose off by 3 degrees: the table looks tilted by about that much."""
        wrong = Capture(capture.scene, camera_pose(53.0))
        assert table_height(wrong).tilt_deg == pytest.approx(3.0, abs=0.5)

    def test_reports_a_second_large_surface(self, capture):
        assert table_height(capture, second_fraction=0.01).other_h == pytest.approx(BOX_Z, abs=0.01)
        assert table_height(capture).other_h is None

    def test_too_little_depth(self, capture):
        empty = Capture(
            Scene(capture.scene.color, np.full((H, W), np.nan, np.float32), FX, FY, PPX, PPY),
            capture.x_root_camera,
        )
        with pytest.raises(ValueError, match="depth points"):
            table_height(empty)


class TestGrid:
    def test_lines_are_multiples_of_the_cell(self):
        grid = Grid(TABLE_Z, 0.05, (0.42, 0.61), (-0.1, 0.1))
        np.testing.assert_allclose(grid.xs, [0.45, 0.5, 0.55, 0.6])
        np.testing.assert_allclose(grid.ys, [-0.1, -0.05, 0, 0.05, 0.1], atol=1e-12)

    def test_extent_is_what_the_view_shows(self, capture):
        full = make_grid(full_view(capture), TABLE_Z, 0.02)
        assert full.h == TABLE_Z
        corners = np.array([[x, y, TABLE_Z] for x in full.x_range for y in full.y_range])
        assert np.hypot(corners[:, 0], corners[:, 1]).max() <= 1.0 * np.sqrt(2)
        zoomed = make_grid(zoom(full_view(capture), (60, 60, 40, 30)), TABLE_Z, 0.01)
        assert full.x_range[0] <= zoomed.x_range[0] < zoomed.x_range[1] <= full.x_range[1]
        assert np.ptp(zoomed.x_range) < np.ptp(full.x_range)

    def test_takes_a_table_estimate(self, capture):
        ws = Workspace()
        view = ws.call(full_view, capture)
        table = ws.call(table_height, capture)
        grid = ws.call(make_grid, view, table)
        assert grid.h == table.h
        assert ws.nodes["grid1"].parents == ("view1", "tableestimate1")

    def test_rejects_too_many_lines_and_an_unseen_plane(self, capture):
        with pytest.raises(ValueError, match="lines per axis"):
            make_grid(full_view(capture), TABLE_Z, 0.001)
        with pytest.raises(ValueError, match="does not show"):
            make_grid(full_view(capture), 2.0, 0.02)  # above the camera


class TestLocate:
    def test_grid_and_depth_agree_on_the_table(self, capture):
        grid = make_grid(full_view(capture), TABLE_Z)
        on_grid = locate_grid(grid, np.array([[0.9, 0.2], [0.5, -0.15]]))
        view = zoom(full_view(capture), (20, 20, 120, 80))
        from_depth = locate_pixels(view, view.project(on_grid.xyz))
        np.testing.assert_allclose(from_depth.xyz, on_grid.xyz, atol=0.01)
        assert from_depth.labels == ("P1", "P2") and from_depth.capture is capture

    def test_box_top_and_missing_depth(self, capture):
        uv = capture.project(np.array([[0.675, 0.0, BOX_Z]]))
        assert locate_pixels(full_view(capture), uv).xyz[0, 2] == pytest.approx(BOX_Z, abs=0.005)
        assert np.isnan(locate_pixels(full_view(capture), np.array([[-5.0, 5.0]])).xyz).all()


class TestCheckPath:
    def test_states(self, capture):
        over = np.array([[0.5, -0.2, 0.6], [0.9, 0.2, 0.6]])  # in the open, between camera and table
        through_box = np.array([[0.5, 0.0, BOX_Z - 0.03], [0.9, 0.0, BOX_Z - 0.03]])
        under = np.array([[0.7, 0.3, 0.1], [0.9, 0.3, 0.1]])  # below the table: hidden
        assert check_path(capture, over).check.status[0] == FREE
        assert check_path(capture, through_box).check.status[0] == BLOCKED
        assert check_path(capture, under).check.status[0] == UNKNOWN

    def test_samples_are_in_the_root_frame(self, capture):
        wp = np.array([[0.5, 0.0, 0.6], [0.9, 0.0, 0.6], [0.9, 0.2, 0.6]])
        plan = check_path(capture, wp)
        np.testing.assert_allclose(plan.check.sample_xyz[0, 0], wp[0], atol=1e-12)
        np.testing.assert_allclose(plan.check.sample_xyz[1, -1], wp[2], atol=1e-12)
        with pytest.raises(ValueError):
            check_path(capture, wp[:1])


class TestRender:
    def test_size_and_input_untouched(self, capture):
        view = zoom(full_view(capture), (40, 30, 80, 60))
        before = capture.scene.color.copy()
        out = render(view, make_grid(view, TABLE_Z))
        assert out.shape == (H, W, 3) and out.dtype == np.uint8
        np.testing.assert_array_equal(capture.scene.color, before)

    def test_grid_is_hidden_on_the_box(self, capture):
        view = full_view(capture)
        plain, gridded = render(view), render(view, make_grid(view, TABLE_Z, 0.02))
        changed = (plain != gridded).any(axis=-1)
        box = box_pixels(capture)
        inside = cv2.erode(box.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)  # anti-aliased edges
        assert changed[~box].any()
        assert not changed[inside].any()

    def test_every_overlay_type(self, capture):
        view = full_view(capture)
        plan = check_path(capture, np.array([[0.5, 0.0, 0.6], [0.9, 0.0, 0.35]]))
        points = locate_pixels(view, np.array([[80.0, 60.0]]))
        out = render(view, make_grid(view, TABLE_Z), table_height(capture), points, plan)
        assert (out != render(view)).any()

    def test_errors(self, capture):
        with pytest.raises(TypeError):
            render(full_view(capture), "grid")
        with pytest.raises(ValueError, match="own capture"):
            render(full_view(make_capture()), table_height(capture))


class TestDescribe:
    def test_text_has_the_numbers_in_cm(self, capture):
        assert "z = 30.0 cm" in table_height(capture).describe()
        assert "2.0 cm cells" in make_grid(full_view(capture), TABLE_Z).describe()
        assert (
            "P1: (90.0, 20.0, 30.0) cm"
            in locate_grid(Grid(TABLE_Z, 0.02, (0, 1), (0, 1)), [[0.9, 0.2]]).describe()
        )
        plan = check_path(capture, np.array([[0.5, 0.0, BOX_Z - 0.03], [0.9, 0.0, BOX_Z - 0.03]]))
        assert "leg 1->2 (40.0 cm): blocked" in plan.describe()


class TestWorkspace:
    def test_records_calls_with_parents(self, capture):
        ws = Workspace()
        view = ws.call(full_view, capture)  # the capture was not added: it becomes "external"
        grid = ws.call(make_grid, view, TABLE_Z, cell=0.05)
        image = ws.call(render, view, grid)
        assert ws.id_of(capture) == "capture1" and ws.nodes["capture1"].call == "external"
        assert ws.nodes["view1"].parents == ("capture1",)
        assert ws.nodes[ws.id_of(image)].parents == ("view1", "grid1")
        assert ws.nodes["grid1"].args == {"arg0": "@view1", "arg1": TABLE_Z, "cell": 0.05}
        assert ws.ancestors(image) == ["view1", "grid1", "capture1"]
        assert ws["grid1"] is grid

    def test_scratch_functions_and_lists_of_parents(self, capture):
        ws = Workspace()
        view = ws.call(full_view, capture)

        def two_points(v: View) -> Points:  # a scratch tool the VLM might write
            return locate_pixels(v, np.array([[60.0, 60.0], [100.0, 60.0]]))

        points = ws.call(two_points, view)
        image = ws.call(lambda v, overlays: render(v, *overlays), view, [points])
        assert ws.nodes["points1"].call == "two_points"
        assert ws.nodes[ws.id_of(image)].parents == ("view1", "points1")
        assert ws.call(lambda: None) is None and len(ws.nodes) == 4

    def test_freshness(self, capture):
        ws = Workspace()
        view = ws.call(full_view, capture)
        by_hand = Grid(TABLE_Z, 0.02, (0.4, 1.0), (-0.3, 0.3))
        ws.add(by_hand)
        assert not ws.is_stale(view)
        ws.moved()
        assert ws.is_stale(view) and ws.is_stale("capture1")
        assert not ws.is_stale(by_hand)  # depends on no capture
        fresh = ws.call(full_view, make_capture())
        plan = ws.call(check_path, ws["capture2"], np.array([[0.5, 0.0, 0.6], [0.9, 0.0, 0.6]]))
        assert not ws.is_stale(fresh) and not ws.is_stale(plan)
        mixed = ws.call(lambda a, b: Points(np.zeros((1, 3)), ("p",)), view, fresh)
        assert ws.is_stale(mixed)  # the oldest capture it depends on wins

    def test_log_is_json(self, capture, tmp_path):
        ws = Workspace()
        view = ws.call(full_view, capture)
        ws.call(check_path, capture, np.array([[0.5, 0.0, 0.6], [0.9, 0.0, 0.6]]))
        ws.call(render, view)
        ws.save_log(tmp_path / "log.jsonl")
        entries = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
        assert [e["id"] for e in entries] == ["capture1", "view1", "plan1", "image1"]
        assert entries[2]["args"]["arg1"] == [[0.5, 0.0, 0.6], [0.9, 0.0, 0.6]]
        assert entries[2]["describe"].startswith("path, 2 waypoints")
        assert entries[3]["describe"] == f"image {W}x{H}"

    def test_survives_pickling(self, capture):
        ws = Workspace()
        view = ws.call(full_view, capture)
        ws.moved()
        loaded = pickle.loads(pickle.dumps(ws))
        assert loaded.id_of(loaded["view1"]) == "view1" and loaded.is_stale("view1")
        grid = loaded.call(make_grid, loaded["view1"], TABLE_Z)
        assert loaded.nodes[loaded.id_of(grid)].parents == ("view1",)
        assert loaded.id_of(view) is None  # the original object is not the unpickled one

    def test_unknown_objects(self):
        ws = Workspace()
        with pytest.raises(KeyError):
            ws.is_stale(TableEstimate(None, 0.3, 0, 0.0, 0.0, np.zeros((1, 1), bool)))  # type: ignore[arg-type]
        assert isinstance(Plan, type)


class TestContactSheetAndFocus:
    def test_contact_sheet_layout(self, capture):
        views = [full_view(capture)] * 4
        sheet = contact_sheet(
            views, Grid(TABLE_Z, 0.05, (0.4, 1.0), (-0.3, 0.3)), labels=list("abcd"), width=80
        )
        assert sheet.shape == (2 * 60, 3 * 80, 3)  # 4 thumbnails, 3 per row, 80 x 60 each
        assert not sheet[60:, 80:].any()  # the two empty slots stay black
        with pytest.raises(ValueError):
            contact_sheet([])

    def test_focus_is_recorded_with_its_views(self, capture):
        ws = Workspace()
        a, b = ws.call(full_view, capture), ws.call(full_view, make_capture(box=False))
        chosen = ws.call(focus, a, b, reason="the box and the table")
        assert chosen.views == (a, b) and "2 view(s): the box" in chosen.describe()
        assert ws.nodes[ws.require_id(chosen)].parents == ("view1", "view2")
        with pytest.raises(ValueError):
            focus()


class TestScan:
    class FakeClient:
        """Records head moves; every capture is the synthetic table scene with the head at its home ticks."""

        def __init__(self, fail_at: int | None = None):
            self.moves: list[tuple[str, float | None, str | None]] = []
            self.captures = 0
            self.fail_at = fail_at
            self.neck = {"home": 1000, "limit_pos": 1500, "limit_neg": 500}

        def head_state(self):
            full = {"home": 1000, "limit_pos": 1500, "limit_neg": 500}
            return {"head_eyes": dict(full), "head_neck": self.neck}

        def head_move(self, name, delta_deg=None, to=None):
            self.moves.append((name, delta_deg, to))
            return {"done_at": float(len(self.moves))}

        def capture(self, after=None):
            self.captures += 1
            if self.captures == self.fail_at:
                raise TimeoutError("server went quiet")
            return "snapshot"

    def test_visits_every_pose_and_returns_home(self, monkeypatch):
        monkeypatch.setattr(scan_module.Capture, "from_snapshot", classmethod(lambda cls, snap, model: snap))
        client = self.FakeClient()
        captures = scan_module.scan(client, model=None, tilts=(0, 15), pans=(20, -20))  # type: ignore[arg-type]
        assert scan_module.scan_poses((0, 15), (20, -20)) == [(0, 20), (0, -20), (15, -20), (15, 20)]
        assert len(captures) == 4
        tilt = sum(d for n, d, _ in client.moves if n == "head_eyes" and d is not None)
        pan = sum(d for n, d, _ in client.moves if n == "head_neck" and d is not None)
        assert (tilt, pan) == (15, 20)  # where the last pose is, relative to home
        assert client.moves[-2:] == [("head_neck", None, "home"), ("head_eyes", None, "home")]

    def test_refuses_to_move_a_servo_without_limits(self):
        client = self.FakeClient()
        client.neck = {"home": None, "limit_pos": None, "limit_neg": None}
        with pytest.raises(RuntimeError, match="head_neck has no recorded home"):
            scan_module.scan(client, model=None)  # type: ignore[arg-type]
        assert client.moves == []

    def test_tilt_only_leaves_the_neck_alone(self, monkeypatch):
        monkeypatch.setattr(scan_module.Capture, "from_snapshot", classmethod(lambda cls, snap, model: snap))
        client = self.FakeClient()
        client.neck = {"home": None, "limit_pos": None, "limit_neg": None}
        assert len(scan_module.scan(client, model=None, pans=(0.0,))) == 3  # type: ignore[arg-type]
        assert {name for name, _, _ in client.moves} == {"head_eyes"}

    def test_goes_home_when_a_capture_fails(self, monkeypatch):
        monkeypatch.setattr(scan_module.Capture, "from_snapshot", classmethod(lambda cls, snap, model: snap))
        client = self.FakeClient(fail_at=2)
        with pytest.raises(TimeoutError):
            scan_module.scan(client, model=None)  # type: ignore[arg-type]
        assert client.moves[-2:] == [("head_neck", None, "home"), ("head_eyes", None, "home")]


DOWN_FRAME = np.column_stack([[0, 1, 0], [1, 0, 0], [0, 0, -1]])  # pinch frame pointing down, jaws along y


def make_poses(capture, ok=(True, False), sweep: bool = False) -> Poses:
    """Two poses from above over the box, jaws 3 cm apart along y (built by hand, no IK); the second unreachable."""
    points = np.array([[0.65, 0.0, BOX_Z + 0.02], [0.7, 0.0, BOX_Z + 0.02]])
    frames = np.repeat(np.eye(4)[None], 2, axis=0)
    frames[:, :3, :3] = DOWN_FRAME
    frames[:, :3, 3] = points + 0.015 * DOWN_FRAME[:, 0]  # the fixed jaw, half the width from the tool point
    extra = {}
    if sweep:
        extra = {
            "targets": points,
            "requested_deg": (-90.0, -60.0),
            "reachable_deg": ((-90.0, -70.0),),
            "target_reachable_deg": (((-90.0, -70.0),), ((-90.0, -60.0),)),
            "back": 0.05,
            "up": 0.05,
        }
    return Poses(
        "right",
        capture,
        points,
        ("at1", "at2"),
        frames,
        ({}, {}),
        np.array([0.0, 0.03]),
        np.array([0.0, 8.0]),
        np.array([0.0, 0.0]),
        np.array(ok),
        -90.0,
        0.03,
        np.array([0.0, 0.0, 1.0]),
        **extra,
    )


def make_no_pitch(capture) -> Poses:
    targets = np.array([[0.65, 0.0, BOX_Z + 0.02], [2.0, 0.0, BOX_Z]])
    empty = np.zeros(0)
    return Poses(
        "right",
        capture,
        np.zeros((0, 3)),
        (),
        np.zeros((0, 4, 4)),
        (),
        empty,
        empty,
        empty,
        empty.astype(bool),
        None,
        0.03,
        np.array([0.0, 0.0, 1.0]),
        targets,
        (-90.0, -60.0),
        (),
        (((-90.0, -60.0),), ()),
        0.05,
        0.05,
    )


class TestPoses:
    def test_describe(self, capture):
        text = make_poses(capture).describe()
        assert text.startswith(
            "right arm pointing down, jaws 3.0 cm apart, jaws closing horizontally: 1/2 reachable"
        )
        assert "at1 ok" in text and "at2 unreachable (3.0 cm short, pitch 8 deg off)" in text
        swept = make_poses(capture, (True, True), sweep=True).describe()
        assert "every target: -90..-70" in swept and "target 2 (70.0, 0.0, 42.0) cm: -90..-60" in swept
        assert "back 5.0 cm along the gripper, up 5.0 cm" in swept
        none = make_no_pitch(capture).describe()
        assert (
            none.startswith("no pitch in -90..-60 reaches every target")
            and "target 2 (200.0, 0.0, 40.0) cm: none" in none
        )

    def test_render(self, capture):
        view = full_view(capture)
        for poses in (make_poses(capture), make_poses(capture, (True, True)), make_no_pitch(capture)):
            assert (render(view, poses) != render(view)).any()

    def test_recorded_and_pickled(self, capture):
        ws = Workspace()
        poses = ws.call(make_poses, capture)
        assert ws.nodes["poses1"].parents == ("capture1",)
        loaded = pickle.loads(pickle.dumps(ws))
        assert loaded["poses1"].capture is loaded["capture1"] and loaded["poses1"].labels == poses.labels
        json.dumps(loaded.log())


def test_reach_pickled_before_pitches_still_loads(capture):
    plan = check_path(capture, np.array([[0.5, 0.0, 0.6], [0.9, 0.0, 0.35]]))
    reach = Reach(plan, "right", ({}, {}), np.zeros(2), np.array([1.0, 9.0]), np.array([True, False]), -90.0)
    state = dict(reach.__dict__)
    state["tilt_deg"] = state.pop("pitch_error_deg")
    del state["pitch_deg"]
    state["down"] = True
    old = Reach.__new__(Reach)
    old.__setstate__(state)
    np.testing.assert_array_equal(old.pitch_error_deg, [1.0, 9.0])
    assert old.pitch_deg == -90.0 and "pointing down" in old.describe()
