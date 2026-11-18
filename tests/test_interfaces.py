"""Contract tests. These must pass before anything is merged to main.

They test the interface (shapes, dtypes, invariants), not output quality,
so they keep passing as stubs are replaced by real implementations.
"""
import numpy as np
import pytest

from dvid.contracts import H, W, check_depth, check_mask, check_rgb
from dvid.geometry import add_to_scene, render, scene_from_rgbd
from dvid.inpainting import inpaint, selective_composite
from dvid.pipeline import CONDITIONS, run_sequence


def make_K():
    f = 0.8 * W
    return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=np.float64)


def make_pose(tx=0.0):
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = tx
    return T


@pytest.fixture
def scene():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    depth = np.full((H, W), 3.0, dtype=np.float32)
    return scene_from_rgbd(rgb, depth, make_K())


def test_scene_from_rgbd(scene):
    assert len(scene) == H * W
    # centre pixel at depth 3 must unproject to (0, 0, 3)
    centre = scene.points[(H // 2) * W + W // 2]
    assert np.allclose(centre, [0, 0, 3], atol=1e-2)


def test_render_contract(scene):
    rgb, holes, depth = render(scene, make_pose(0.1))
    check_rgb(rgb)
    check_mask(holes)
    check_depth(depth)
    assert np.all(depth[holes] == 0), "holes must have invalid (0) depth"


def test_render_identity_has_no_holes(scene):
    _, holes, _ = render(scene, make_pose(0.0))
    assert holes.mean() < 0.01, "rendering the source view should leave (almost) no holes"


def test_add_to_scene_is_pure(scene):
    rgb, holes, _ = render(scene, make_pose(0.1))
    n_before = len(scene)
    new = add_to_scene(scene, rgb, holes, make_pose(0.1))
    assert len(scene) == n_before, "add_to_scene must not mutate its input"
    assert len(new) >= n_before


def test_inpaint_contract(scene):
    rgb, holes, _ = render(scene, make_pose(0.1))
    raw = inpaint(rgb, holes, seed=0)
    check_rgb(raw)
    assert np.array_equal(raw, inpaint(rgb, holes, seed=0)), "same seed must give same output"


def test_selective_composite_preserves_visible(scene):
    rgb, holes, _ = render(scene, make_pose(0.1))
    raw = inpaint(rgb, holes)
    out = selective_composite(rgb, raw, holes)
    assert np.array_equal(out[~holes], rgb[~holes]), "visible pixels must be untouched"
    assert np.array_equal(out[holes], raw[holes])


@pytest.mark.parametrize("condition", CONDITIONS)
def test_pipeline_runs(scene, condition):
    poses = [make_pose(tx) for tx in (0.0, 0.05, 0.1)]
    frames = run_sequence(scene, poses, condition)
    assert len(frames) == 3
    for f in frames:
        check_rgb(f["rgb"])
        check_mask(f["hole_mask"])
