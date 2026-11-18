"""Scale alignment must recover a known scale on a synthetic scene (step 6)."""
import cv2
import numpy as np
import pytest

from dvid.alignment import estimate_clip_scale, estimate_scale, with_scaled_translation
from dvid.contracts import H, W
from dvid.geometry import pose_from_center, render, scene_from_rgbd


def textured_image(seed=0):
    rng = np.random.default_rng(seed)
    coarse = cv2.resize(rng.integers(0, 256, (36, 64, 3), dtype=np.uint8), (W, H), interpolation=cv2.INTER_CUBIC)
    fine = rng.integers(-40, 40, (H, W, 3))
    return np.clip(coarse.astype(int) + fine, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def setup():
    K = np.array([[0.8 * W, 0, W / 2], [0, 0.8 * W, H / 2], [0, 0, 1]], dtype=np.float64)
    rgb = textured_image()
    depth = np.full((H, W), 4.0, np.float32)
    depth[150:420, 300:700] = 2.5  # a box in front of the wall
    scene = scene_from_rgbd(rgb, depth, K)
    return K, rgb, depth, scene


@pytest.mark.parametrize("s_true", [0.5, 7.5, 60.0])
def test_recovers_known_scale(setup, s_true):
    K, rgb, depth, scene = setup
    T_rel_metric = pose_from_center([0.25, 0.02, 0.1])       # true motion in metres
    tgt_rgb, _, _ = render(scene, T_rel_metric)

    # Dataset poses: arbitrary world frame, translation in unknown units (= metres / s_true)
    a = np.radians(20)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    T_src = pose_from_center([1.0, -0.5, 2.0], R)
    T_tgt = with_scaled_translation(T_rel_metric, 1.0 / s_true) @ T_src

    r = estimate_scale(rgb, tgt_rgb, depth, K, T_src, T_tgt)
    assert r["n_used"] > 100
    assert abs(r["scale"] / s_true - 1) < 0.03, r
    assert r["median_err_px"] < 1.0
    assert r["median_err_px"] < r["median_err_px_unscaled"] or np.isclose(s_true, 1.0)


def test_clip_scale_ignores_weak_pairs():
    results = [
        {"scale": 10.0, "trans_px": 40, "n_used": 200},
        {"scale": 10.4, "trans_px": 25, "n_used": 150},
        {"scale": 3.0, "trans_px": 2, "n_used": 300},    # too little motion: ignored
        {"scale": 99.0, "trans_px": 50, "n_used": 5},    # too few points: ignored
        {"scale": np.nan, "trans_px": np.nan, "n_used": 0},
    ]
    s, n = estimate_clip_scale(results)
    assert n == 2 and np.isclose(s, 10.2)
