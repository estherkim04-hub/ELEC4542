"""Add border masks to sequences that were exported before classify_holes existed.

    python scripts/split_holes.py --runs runs

For every <runs>/<sequence>/A_warp/seed0/ that has hole_XXXX.png, depth_XXXX.npy
and poses.json, writes border_XXXX.png (white = hole whose content was OUTSIDE the
source view) and hole_types.json (per-frame hole / border / disocclusion fractions).
Disocclusion = hole AND NOT border. Needs the depth files, so run it on the full
runs folder, not runs_share.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dvid.geometry import classify_holes  # noqa: E402


def split_sequence(seq_dir):
    with open(os.path.join(seq_dir, "poses.json")) as f:
        cams = json.load(f)
    K = np.array(cams["K"], dtype=np.float64)
    rows = []
    for i, T in enumerate(cams["poses"]):
        hole_p = os.path.join(seq_dir, f"hole_{i:04d}.png")
        depth_p = os.path.join(seq_dir, f"depth_{i:04d}.npy")
        if not (os.path.exists(hole_p) and os.path.exists(depth_p)):
            return None
        holes = np.array(Image.open(hole_p)) > 127
        depth = np.load(depth_p).astype(np.float32)
        border, disocc = classify_holes(holes, depth, K, np.array(T, dtype=np.float64))
        Image.fromarray(border.astype(np.uint8) * 255).save(os.path.join(seq_dir, f"border_{i:04d}.png"))
        rows.append({"frame": i, "hole_fraction": float(holes.mean()),
                     "border_fraction": float(border.mean()), "disocclusion_fraction": float(disocc.mean())})
    with open(os.path.join(seq_dir, "hole_types.json"), "w") as f:
        json.dump(rows, f, indent=1)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args()
    seqs = sorted(glob.glob(os.path.join(args.runs, "*", "A_warp", "seed0")))
    assert seqs, f"no sequences under {args.runs}"
    for s in seqs:
        rows = split_sequence(s)
        name = s.split(os.sep)[-3]
        if rows is None:
            print(f"{name}: missing hole or depth files, skipped")
            continue
        last = rows[-1]
        print(f"{name}: final frame holes {last['hole_fraction']:.1%} = border {last['border_fraction']:.1%}"
              f" + disocclusion {last['disocclusion_fraction']:.1%}")


if __name__ == "__main__":
    main()
