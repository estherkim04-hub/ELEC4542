"""Member 1: depth, scene construction, rendering, scene update.

Everything below is a STUB that returns correctly shaped dummy data, so that
Member 2 can build and test against these functions from day one.
Replace each body with the real implementation; keep the signatures and the
contract checks.
"""
import numpy as np

from .contracts import H, W, check_depth, check_K, check_mask, check_pose, check_rgb
from .types import Scene


def scene_from_rgbd(rgb, depth, K):
    """Lift a source RGB-D image into a Scene (world = source camera frame).

    STUB: real implementation unprojects every valid pixel with K.
    Kept real-ish here because it is short and useful for testing.
    """
    check_rgb(rgb)
    check_depth(depth)
    check_K(K)
    v, u = np.nonzero(depth > 0)
    z = depth[v, u].astype(np.float64)
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    return Scene(points=pts, colors=rgb[v, u].copy(), K=K)


def render(scene, T):
    """Render `scene` from world-to-camera pose `T`.

    Returns:
        warped_rgb (H,W,3) uint8, hole_mask (H,W) bool, depth_map (H,W) float32 metres.

    STUB: grey frame with a rectangular "disocclusion" on the left edge whose
    width grows with the camera's x-translation, so downstream code sees
    plausible, pose-dependent holes.
    """
    assert isinstance(scene, Scene)
    check_pose(T)

    warped_rgb = np.full((H, W, 3), 128, dtype=np.uint8)
    depth_map = np.full((H, W), 3.0, dtype=np.float32)
    hole_mask = np.zeros((H, W), dtype=bool)

    hole_w = int(min(W // 2, abs(T[0, 3]) * 400))
    if hole_w > 0:
        hole_mask[:, :hole_w] = True
        warped_rgb[hole_mask] = 0
        depth_map[hole_mask] = 0.0

    check_rgb(warped_rgb, "warped_rgb")
    check_mask(hole_mask, "hole_mask")
    check_depth(depth_map, "depth_map")
    return warped_rgb, hole_mask, depth_map


def add_to_scene(scene, completed_rgb, hole_mask, T):
    """Add the inpainted pixels (hole_mask == True) of `completed_rgb`, seen from
    pose `T`, to the scene. Returns a NEW Scene; never mutates `scene`.

    Real implementation: run DA3 on completed_rgb, fit scale+shift to the
    rendered depth in a ring around the holes, reject unreliable points,
    transform to world, append.

    STUB: returns an unchanged copy.
    """
    assert isinstance(scene, Scene)
    check_rgb(completed_rgb, "completed_rgb")
    check_mask(hole_mask, "hole_mask")
    check_pose(T)
    return Scene(points=scene.points.copy(), colors=scene.colors.copy(), K=scene.K)
