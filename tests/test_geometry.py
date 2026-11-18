"""Behavioural tests for the renderer and camera helpers (Member 1)."""
import numpy as np
import pytest

from dvid.contracts import H, W
from dvid.geometry import (
    camera_center, camera_path, normalized_displacement, pose_from_center,
    render, rotation_angle_deg, scene_from_rgbd,
)

F = 0.8 * W


@pytest.fixture
def K():
    return np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]], dtype=np.float64)


@pytest.fixture
def rgb():
    return np.random.default_rng(0).integers(0, 256, (H, W, 3), dtype=np.uint8)


def plane(z):
    return np.full((H, W), z, dtype=np.float32)


def test_identity_reproduces_source(rgb, K):
    scene = scene_from_rgbd(rgb, plane(3.0), K)
    out, holes, depth = render(scene, np.eye(4))
    assert not holes.any()
    assert np.array_equal(out, rgb)
    assert np.allclose(depth, 3.0)


def test_lateral_move_shifts_image(rgb, K):
    """Camera moves right by 0.15 m in front of a plane at 3 m: the image must
    shift LEFT by fx * 0.15 / 3 = 40.96 px, and new content appears on the right."""
    scene = scene_from_rgbd(rgb, plane(3.0), K)
    out, holes, _ = render(scene, pose_from_center([0.15, 0, 0]))
    shift = 41
    assert np.array_equal(out[:, : W - shift - 2], rgb[:, shift : W - 2])
    assert holes[:, -shift:].all() and not holes[:, : W - shift - 2].any()


def test_foreground_occludes_and_disoccludes(rgb, K):
    """Square at 2 m in front of a wall at 4 m. Moving right reveals wall on the
    square's right side; the square must stay in front of the wall."""
    depth = plane(4.0)
    y0, y1, x0, x1 = 188, 388, 412, 612
    depth[y0:y1, x0:x1] = 2.0
    scene = scene_from_rgbd(rgb, depth, K)
    out, holes, rdepth = render(scene, pose_from_center([0.2, 0, 0]))

    fg_shift = F * 0.2 / 2.0   # 81.9 px
    bg_shift = F * 0.2 / 4.0   # 41.0 px
    mid_y = (y0 + y1) // 2
    # centre of the square, shifted, still shows the square at 2 m
    cx = int((x0 + x1) / 2 - fg_shift)
    assert np.isclose(rdepth[mid_y, cx], 2.0)
    # a band just right of the shifted square is a disocclusion
    band = slice(int(x1 - fg_shift) + 3, int(x1 - bg_shift) - 3)
    assert holes[mid_y, band].all()
    # and left of the square there is no hole (background was already visible)
    assert not holes[mid_y, int(x0 - fg_shift) - 30 : int(x0 - fg_shift) - 5].any()


def test_dolly_forward_has_no_cracks(rgb, K):
    """Moving 0.3 m towards a plane at 3 m magnifies it ~1.11x: no holes allowed."""
    scene = scene_from_rgbd(rgb, plane(3.0), K)
    _, holes, _ = render(scene, pose_from_center([0, 0, 0.3]))
    assert holes.mean() < 0.001


def test_camera_path_endpoints():
    poses = camera_path("lateral", scene_median_depth=3.0, d_norm=0.1, n_frames=49)
    assert len(poses) == 49
    assert np.allclose(poses[0], np.eye(4))
    assert np.allclose(camera_center(poses[-1]), [0.3, 0, 0])
    assert np.isclose(normalized_displacement(poses[0], poses[-1], plane(3.0)), 0.1)


def test_orbit_looks_at_centre():
    poses = camera_path("orbit", scene_median_depth=3.0, d_norm=0.2, n_frames=9)
    T = poses[-1]
    target_cam = T[:3, :3] @ np.array([0, 0, 3.0]) + T[:3, 3]
    assert abs(target_cam[0]) < 1e-9 and target_cam[2] > 0  # stays centred, in front
    # arc length 0.2 * 3 m = 0.6 m on radius 3 m -> 0.2 rad = 11.46 deg
    assert np.isclose(rotation_angle_deg(poses[0], T), np.degrees(0.2))
