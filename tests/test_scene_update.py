"""Condition D: add_to_scene must fill holes with correctly placed points, so a
later render re-uses them instead of needing new inpainting."""
import cv2
import numpy as np
import pytest

from dvid import geometry
from dvid.contracts import H, W
from dvid.geometry import add_to_scene, pose_from_center, render, scene_from_rgbd
from dvid.types import Scene

F = 0.8 * W
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]], dtype=np.float64)
BOX = (150, 420, 300, 700)  # y0, y1, x0, x1 of a box at 2.5 m in front of a 4 m wall


def texture(seed):
    rng = np.random.default_rng(seed)
    c = cv2.resize(rng.integers(0, 256, (36, 64, 3), dtype=np.uint8), (W, H), interpolation=cv2.INTER_CUBIC)
    return np.clip(c.astype(int) + rng.integers(-30, 30, (H, W, 3)), 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def world():
    """Source scene (what the source photo sees) and a ground-truth scene that
    also contains the wall behind the box and beyond the image edges."""
    wall_tex, box_tex = texture(1), texture(2)
    y0, y1, x0, x1 = BOX
    src_rgb = wall_tex.copy()
    src_rgb[y0:y1, x0:x1] = box_tex[y0:y1, x0:x1]
    src_depth = np.full((H, W), 4.0, np.float32)
    src_depth[y0:y1, x0:x1] = 2.5
    src = scene_from_rgbd(src_rgb, src_depth, K)

    # ground truth: a wide wall (3x the field of view) + the box
    Kw = K.copy()
    Kw[0, 0] /= 3
    Kw[1, 1] /= 3
    wall = scene_from_rgbd(wall_tex, np.full((H, W), 4.0, np.float32), Kw, drop_depth_edges=False)
    box_d = np.zeros((H, W), np.float32)
    box_d[y0:y1, x0:x1] = 2.5
    box = scene_from_rgbd(src_rgb, box_d, K, drop_depth_edges=False)
    gt = Scene(points=np.concatenate([wall.points, box.points]),
               colors=np.concatenate([wall.colors, box.colors]), K=K)
    return src, gt


@pytest.fixture(autouse=True)
def no_estimator():
    geometry.set_depth_estimator(None)
    yield
    geometry.set_depth_estimator(None)


def inpaint_perfectly(gt, T, holes, warped):
    """Stand-in for a perfect inpainter: ground-truth colours inside the holes."""
    truth, _, _ = render(gt, T, hole_dilate=0)
    out = warped.copy()
    out[holes] = truth[holes]
    return out, truth


def test_reuse_fills_holes_at_same_pose(world):
    src, gt = world
    T = pose_from_center([0.3, 0, 0])
    warped, holes, depth = render(src, T)
    completed, truth = inpaint_perfectly(gt, T, holes, warped)
    new, stats = add_to_scene(src, completed, holes, T, rendered_depth=depth, return_stats=True)

    assert stats["mode"] == "fill" and stats["added"] > 0.9 * stats["holes"]
    again, holes2, _ = render(new, T)
    assert holes2.mean() < 0.15 * holes.mean()          # most holes now re-rendered
    reused = holes & ~holes2
    err = np.abs(again[reused].astype(int) - truth[reused].astype(int)).mean()
    assert err < 12, err                                # and with the right colours


def test_added_points_sit_on_the_background(world):
    src, gt = world
    T = pose_from_center([0.3, 0, 0])
    warped, holes, depth = render(src, T)
    completed, _ = inpaint_perfectly(gt, T, holes, warped)
    new = add_to_scene(src, completed, holes, T, rendered_depth=depth)
    added = new.points[len(src):]
    assert np.allclose(np.median(added[:, 2]), 4.0, rtol=0.02)   # the wall, not the box


def test_reuse_carries_to_the_next_frame(world):
    """After updating at T1, the next pose shows fewer holes, and the re-used
    region matches the ground truth there (no re-inpainting needed)."""
    src, gt = world
    T1, T2 = pose_from_center([0.25, 0, 0]), pose_from_center([0.3, 0, 0])
    warped, holes, depth = render(src, T1)
    completed, _ = inpaint_perfectly(gt, T1, holes, warped)
    new = add_to_scene(src, completed, holes, T1, rendered_depth=depth)

    _, holes_without, _ = render(src, T2)
    with_reuse, holes_with, _ = render(new, T2)
    assert holes_with.mean() < 0.5 * holes_without.mean()
    truth2, _, _ = render(gt, T2, hole_dilate=0)
    reused = holes_without & ~holes_with
    err = np.abs(with_reuse[reused].astype(int) - truth2[reused].astype(int)).mean()
    assert err < 15, err


def test_estimator_is_aligned_to_scene(world):
    """A depth model with the wrong scale and offset is corrected by the ring fit."""
    src, gt = world
    T = pose_from_center([0.3, 0, 0])
    warped, holes, depth = render(src, T)
    completed, _ = inpaint_perfectly(gt, T, holes, warped)
    _, _, true_depth = render(gt, T, hole_dilate=0)
    true_depth = np.where(true_depth > 0, true_depth, 4.0)
    geometry.set_depth_estimator(lambda rgb: 0.5 * true_depth + 0.3)
    new, stats = add_to_scene(src, completed, holes, T, rendered_depth=depth, return_stats=True)
    assert stats["mode"] == "estimator"
    assert np.allclose(np.median(new.points[len(src):, 2]), 4.0, rtol=0.02)


def test_no_holes_returns_unchanged_copy(world):
    src, _ = world
    rgb, holes, depth = render(src, np.eye(4), hole_dilate=0)
    holes[:] = False
    new = add_to_scene(src, rgb, holes, np.eye(4), rendered_depth=depth)
    assert len(new) == len(src) and new.points is not src.points
