"""Runs one camera trajectory under one experimental condition (A-D).

This is the only place where the conditions differ, so the controlled
comparison is visible in ~15 lines.
"""
import time

from .geometry import add_to_scene, render
from .inpainting import inpaint, selective_composite

CONDITIONS = ("A_warp", "B_full", "C_selective", "D_reuse")


def run_sequence(scene, poses, condition, seed=0):
    """Generate one frame per pose.

    Returns a list of dicts: rgb, hole_mask, depth, timing (seconds per stage).
    """
    assert condition in CONDITIONS, f"unknown condition {condition!r}"
    frames = []
    for T in poses:
        t = {}
        t0 = time.perf_counter()
        warped, holes, depth = render(scene, T)
        t["render"] = time.perf_counter() - t0

        if condition == "A_warp" or not holes.any():
            out = warped
        else:
            t0 = time.perf_counter()
            raw = inpaint(warped, holes, seed=seed)
            t["inpaint"] = time.perf_counter() - t0
            out = raw if condition == "B_full" else selective_composite(warped, raw, holes)

        if condition == "D_reuse" and holes.any():
            t0 = time.perf_counter()
            scene = add_to_scene(scene, out, holes, T, rendered_depth=depth)
            t["scene_update"] = time.perf_counter() - t0

        frames.append({"rgb": out, "hole_mask": holes, "depth": depth,
                       "hole_fraction": float(holes.mean()), "timing": t})
    return frames
