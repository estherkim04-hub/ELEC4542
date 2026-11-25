"""classify_holes: border holes (outside the source view) vs disocclusions."""
import numpy as np

from dvid.contracts import H, W
from dvid.geometry import classify_holes, pose_from_center, render, scene_from_rgbd

F = 0.8 * W
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]], dtype=np.float64)
Y0, Y1, X0, X1 = 150, 420, 300, 700   # box at 2.5 m in front of a wall at 4 m


def box_scene():
    rgb = np.random.default_rng(0).integers(0, 256, (H, W, 3), dtype=np.uint8)
    depth = np.full((H, W), 4.0, np.float32)
    depth[Y0:Y1, X0:X1] = 2.5
    return scene_from_rgbd(rgb, depth, K)


def test_lateral_move_splits_border_and_disocclusion():
    T = pose_from_center([0.3, 0, 0])            # camera moves right
    _, holes, depth = render(box_scene(), T)
    border, disocc = classify_holes(holes, depth, K, T)
    assert not (border & disocc).any() and np.array_equal(border | disocc, holes)

    mid = (Y0 + Y1) // 2
    edge = int(F * 0.3 / 4.0)                    # ~61 px of new content on the right
    assert border[:, W - edge + 3:].all()        # right strip = border
    fg_shift, bg_shift = F * 0.3 / 2.5, F * 0.3 / 4.0
    band = slice(int(X1 - fg_shift) + 4, int(X1 - bg_shift) - 4)
    assert disocc[mid, band].all()               # beside the box = disocclusion
    assert not border[mid, band].any()


def test_dolly_back_gives_border_on_all_sides():
    T = pose_from_center([0, 0, -0.5])           # step back: the view widens
    _, holes, depth = render(box_scene(), T)
    border, disocc = classify_holes(holes, depth, K, T)
    for strip in (border[:3, :], border[-3:, :], border[:, :3], border[:, -3:]):
        assert strip.mean() > 0.95
    assert border.sum() > 5 * disocc.sum()


def test_no_holes_no_classes():
    _, holes, depth = render(box_scene(), np.eye(4), hole_dilate=0)
    holes[:] = False
    border, disocc = classify_holes(holes, depth, K, np.eye(4))
    assert not border.any() and not disocc.any()
