"""Render camera paths from DA3 outputs and export warped frames + hole masks
(Condition A) in the CONVENTIONS.md layout, for Member 2's inpainting.

    python scripts/export_renders.py --depth-dir data/depth --out runs \
        --paths lateral orbit --d-norm 0.05 0.1 0.2 --frames 25

Input:  <depth-dir>/<name>/{rgb.jpg, depth.npy, K.npy}   (from run_da3.py)
Output: <out>/photo_<name>_<path>_d<d_norm>/A_warp/seed0/
            frame_0000.png  hole_0000.png  depth_0000.npy  poses.json  timing.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dvid.geometry import camera_path, median_depth, render, scene_from_rgbd  # noqa: E402


def export_sequence(scene, poses, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    timing = []
    for i, T in enumerate(poses):
        t0 = time.perf_counter()
        rgb, holes, depth = render(scene, T)
        timing.append({"render": time.perf_counter() - t0, "hole_fraction": float(holes.mean())})
        Image.fromarray(rgb).save(os.path.join(out_dir, f"frame_{i:04d}.png"))
        Image.fromarray(holes.astype(np.uint8) * 255).save(os.path.join(out_dir, f"hole_{i:04d}.png"))
        np.save(os.path.join(out_dir, f"depth_{i:04d}.npy"), depth)
    with open(os.path.join(out_dir, "poses.json"), "w") as f:
        json.dump({"K": scene.K.tolist(), "poses": [T.tolist() for T in poses]}, f)
    with open(os.path.join(out_dir, "timing.json"), "w") as f:
        json.dump(timing, f, indent=1)
    return timing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth-dir", required=True)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--paths", nargs="+", default=["lateral"], choices=["lateral", "dolly", "orbit"])
    ap.add_argument("--d-norm", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    ap.add_argument("--frames", type=int, default=25, help="use 4n+1")
    args = ap.parse_args()

    names = sorted(d for d in os.listdir(args.depth_dir)
                   if os.path.exists(os.path.join(args.depth_dir, d, "depth.npy")))
    assert names, f"no DA3 outputs found in {args.depth_dir}"

    for name in names:
        src = os.path.join(args.depth_dir, name)
        rgb = np.asarray(Image.open(os.path.join(src, "rgb.jpg")).convert("RGB"), dtype=np.uint8)
        depth = np.load(os.path.join(src, "depth.npy")).astype(np.float32)
        K = np.load(os.path.join(src, "K.npy")).astype(np.float64)
        scene = scene_from_rgbd(rgb, depth, K)
        md = median_depth(depth)
        for kind in args.paths:
            for d in args.d_norm:
                poses = camera_path(kind, md, d, args.frames)
                out_dir = os.path.join(args.out, f"photo_{name}_{kind}_d{d:g}", "A_warp", "seed0")
                timing = export_sequence(scene, poses, out_dir)
                print(f"{out_dir}: final-frame holes {timing[-1]['hole_fraction']:.1%}, "
                      f"{np.mean([t['render'] for t in timing]):.2f}s/frame")


if __name__ == "__main__":
    main()
