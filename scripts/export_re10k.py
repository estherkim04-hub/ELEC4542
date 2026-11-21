"""Render RealEstate10K clips along their REAL camera paths (Condition A), with
the matching real frames as ground truth, in the CONVENTIONS.md runs/ layout.

    python scripts/export_re10k.py --re10k data/re10k --depth-dir data/re10k_depth --out runs

For each clip <id> that has data/re10k/<id>/ (from re10k_extract.py) and
data/re10k_depth/<id>/depth.npy (from run_da3.py on its source frame):

1. Estimate (or reuse) the clip's dataset-to-metres scale -> data/re10k/<id>/scale.json
2. For each d_norm band, walk the real trajectory until the camera has moved
   that far (rotation < --max-rot), and sample --frames frames evenly.
3. Render the source frame into each sampled view.

Output per clip and band:  runs/re10k_<id>_d<band>/
    A_warp/seed0/frame_XXXX.png  hole_XXXX.png  depth_XXXX.npy  poses.json  timing.json
    gt/frame_XXXX.jpg            the REAL frame at the same camera (ground truth)
    meta.json                    clip id, scale, frame indices, timestamps, d_norm per frame
Frame 0000 is always the source frame itself.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dvid.alignment import clip_scale, with_scaled_translation  # noqa: E402
from dvid.geometry import scene_from_rgbd  # noqa: E402
from dvid.re10k import candidate_targets, Clip, select_sequence  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from export_renders import export_sequence  # noqa: E402


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def get_scale(clip_dir, frame, poses, K, depth, src, max_rot, recompute=False):
    path = os.path.join(clip_dir, "scale.json")
    if os.path.exists(path) and not recompute:
        with open(path) as f:
            saved = json.load(f)
        if saved.get("src") == src:
            return saved["scale"]
    ts = np.zeros(len(poses), np.int64)
    cands = [j for j, _, _ in candidate_targets(Clip("", "", ts, np.zeros((len(poses), 4)), poses), src, max_rot)]
    s, n_good, per_pair = clip_scale(frame, poses, K, depth, src, cands)
    if not np.isfinite(s):
        return None
    with open(path, "w") as f:
        json.dump({"src": src, "scale": s, "pairs_used": n_good, "per_pair": per_pair}, f, indent=1, default=float)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re10k", default="data/re10k")
    ap.add_argument("--depth-dir", default="data/re10k_depth")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--clips", nargs="*", help="clip ids (default: all with depth)")
    ap.add_argument("--d-norm", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    ap.add_argument("--frames", type=int, default=25, help="frames per sequence (use 4n+1)")
    ap.add_argument("--max-rot", type=float, default=15.0)
    ap.add_argument("--src", type=int, default=0, help="source frame index")
    ap.add_argument("--recompute-scale", action="store_true")
    args = ap.parse_args()

    clips = args.clips or sorted(d for d in os.listdir(args.depth_dir)
                                 if os.path.exists(os.path.join(args.depth_dir, d, "depth.npy")))
    assert clips, f"no depth found in {args.depth_dir}"

    for cid in clips:
        clip_dir = os.path.join(args.re10k, cid)
        cams_path = os.path.join(clip_dir, "cameras.npz")
        if not os.path.exists(cams_path):
            print(f"{cid}: no {cams_path}, skipping")
            continue
        cams = np.load(cams_path)
        ts, Ks, poses = cams["timestamps_us"], cams["K"], cams["poses"]
        frame = lambda i, d=clip_dir, t=ts: load_rgb(os.path.join(d, "frames", f"{t[i]}.jpg"))  # noqa: E731
        depth = np.load(os.path.join(args.depth_dir, cid, "depth.npy")).astype(np.float32)
        src, K = args.src, Ks[args.src]

        s = get_scale(clip_dir, frame, poses, K, depth, src, args.max_rot, args.recompute_scale)
        if s is None:
            print(f"{cid}: scale could not be estimated (too few matches or too little motion), skipping")
            continue
        print(f"{cid}: scale {s:.4g}")

        scene = scene_from_rgbd(frame(src), depth, K)
        T_src_inv = np.linalg.inv(poses[src])
        for band in args.d_norm:
            sel = select_sequence(poses, src, s, depth, band, args.frames, args.max_rot)
            if sel is None:
                print(f"  d{band:g}: camera never moves that far within {args.max_rot} deg "
                      f"(or too few frames), skipping")
                continue
            idx, d_norms = sel
            seq_dir = os.path.join(args.out, f"re10k_{cid}_d{band:g}")
            rel_poses = [with_scaled_translation(poses[j] @ T_src_inv, s) for j in idx]
            timing = export_sequence(scene, rel_poses, os.path.join(seq_dir, "A_warp", "seed0"))

            gt_dir = os.path.join(seq_dir, "gt")
            os.makedirs(gt_dir, exist_ok=True)
            for k, j in enumerate(idx):  # copy the extracted JPEG as-is (no re-encoding)
                shutil.copy(os.path.join(clip_dir, "frames", f"{ts[j]}.jpg"), os.path.join(gt_dir, f"frame_{k:04d}.jpg"))
            with open(os.path.join(seq_dir, "meta.json"), "w") as f:
                json.dump({"clip_id": cid, "scale": s, "src_index": int(src), "band": band,
                           "frame_indices": [int(j) for j in idx],
                           "timestamps_us": [int(ts[j]) for j in idx],
                           "d_norm": d_norms}, f, indent=1)
            print(f"  d{band:g}: frames {idx[0]}..{idx[-1]}, final d_norm {d_norms[-1]:.3f}, "
                  f"final holes {timing[-1]['hole_fraction']:.1%} -> {seq_dir}")


if __name__ == "__main__":
    main()
