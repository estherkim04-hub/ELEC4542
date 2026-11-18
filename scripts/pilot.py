"""Week 1 milestone: one RealEstate10K clip, start to finish.

    python scripts/pilot.py --clip-dir data/re10k/<clip_id> --depth data/depth/<src_timestamp>/depth.npy

1. Pick target frames whose rotation from the source is < --max-rot degrees.
2. Estimate the dataset-to-depth scale on several of them; take a robust median.
3. Render the source frame into one target view with the aligned pose.
4. Compare with the real target frame on visible pixels (PSNR, SSIM) and report
   reprojection error before/after alignment and the normalised displacement.

Writes <clip-dir>/pilot/pilot.png (source | render, holes magenta |
real target | visible-pixel error) and pilot.json.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dvid.alignment import (  # noqa: E402
    _lift, estimate_clip_scale, estimate_scale, match_features, reprojection_errors, with_scaled_translation,
)
from dvid.geometry import normalized_displacement, render, scene_from_rgbd  # noqa: E402
from dvid.re10k import Clip, candidate_targets  # noqa: E402


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def masked_psnr(a, b, mask):
    mse = np.mean((a[mask].astype(np.float64) - b[mask].astype(np.float64)) ** 2)
    return float(10 * np.log10(255.0 ** 2 / mse)) if mse > 0 else float("inf")


def masked_ssim(a, b, mask):
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return None
    _, smap = structural_similarity(a, b, channel_axis=2, full=True, data_range=255)
    return float(smap.mean(axis=2)[mask].mean())


def error_heatmap(a, b, mask):
    err = np.abs(a.astype(np.int16) - b.astype(np.int16)).mean(axis=2)
    e = np.clip(err / 60.0, 0, 1)
    vis = np.stack([255 * e, 255 * (1 - np.abs(2 * e - 1)), 255 * (1 - e)], -1).astype(np.uint8)
    vis[~mask] = 0
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-dir", required=True)
    ap.add_argument("--depth", required=True, help="DA3 depth.npy for the SOURCE frame")
    ap.add_argument("--src", type=int, default=0, help="source frame index")
    ap.add_argument("--tgt", type=int, default=None, help="target index to render (default: furthest valid)")
    ap.add_argument("--max-rot", type=float, default=15.0)
    ap.add_argument("--n-scale-pairs", type=int, default=8)
    args = ap.parse_args()

    cams = np.load(os.path.join(args.clip_dir, "cameras.npz"))
    ts, Ks, poses = cams["timestamps_us"], cams["K"], cams["poses"]
    frame = lambda i: load_rgb(os.path.join(args.clip_dir, "frames", f"{ts[i]}.jpg"))  # noqa: E731

    clip = Clip("pilot", "", ts, np.zeros((len(ts), 4)), poses)
    cands = candidate_targets(clip, args.src, args.max_rot)
    assert cands, "no target frames within the rotation limit"
    print(f"{len(cands)} candidate targets within {args.max_rot} deg of frame {args.src}")

    src_rgb = frame(args.src)
    depth = np.load(args.depth).astype(np.float32)
    K = Ks[args.src]
    T_src = poses[args.src]

    # --- 2. scale on several evenly spaced targets ---------------------------
    pick = np.linspace(len(cands) // 4, len(cands) - 1, min(args.n_scale_pairs, len(cands))).astype(int)
    results = []
    for k in np.unique(pick):
        j = cands[k][0]
        r = estimate_scale(src_rgb, frame(j), depth, K, T_src, poses[j])
        r["target"] = int(j)
        results.append(r)
        print(f"  target {j:4d}: scale {r['scale']:.4g}  err {r['median_err_px']:.2f}px "
              f"(unscaled {r['median_err_px_unscaled']:.1f}px)  matches {r['n_used']}  motion {r['trans_px']:.0f}px")
    s, n_good = estimate_clip_scale(results)
    assert np.isfinite(s), "no well-constrained pair; try another source frame or clip"
    print(f"clip scale = {s:.4g} (median of {n_good} well-constrained pairs)")

    # --- 3. render into one target --------------------------------------------
    j = args.tgt if args.tgt is not None else cands[-1][0]
    T_rel = with_scaled_translation(poses[j] @ np.linalg.inv(T_src), s)
    scene = scene_from_rgbd(src_rgb, depth, K)
    rendered, holes, _ = render(scene, T_rel)
    tgt_rgb = frame(j)
    vis = ~holes

    # --- 4. metrics -----------------------------------------------------------
    pa, pb = match_features(src_rgb, tgt_rgb, K)
    X, keep = _lift(pa, depth, K)
    T_rel_ds = poses[j] @ np.linalg.inv(T_src)
    err_aligned = float(np.median(reprojection_errors(X, pb[keep], K, T_rel_ds, s))) if len(X) else None
    err_unscaled = float(np.median(reprojection_errors(X, pb[keep], K, T_rel_ds, 1.0))) if len(X) else None
    report = {
        "source": int(args.src), "target": int(j), "clip_scale": s, "pairs_used": n_good,
        "d_norm": normalized_displacement(np.eye(4), T_rel, depth),
        "hole_fraction": float(holes.mean()),
        "visible_psnr": masked_psnr(rendered, tgt_rgb, vis),
        "visible_ssim": masked_ssim(rendered, tgt_rgb, vis),
        "reproj_err_px_aligned": err_aligned,
        "reproj_err_px_unscaled": err_unscaled,
        "per_pair": results,
    }
    out_dir = os.path.join(args.clip_dir, "pilot")
    os.makedirs(out_dir, exist_ok=True)
    shown = rendered.copy()
    shown[holes] = [255, 0, 255]
    grid = np.concatenate([np.concatenate([src_rgb, shown], 1),
                           np.concatenate([tgt_rgb, error_heatmap(rendered, tgt_rgb, vis)], 1)], 0)
    Image.fromarray(grid).save(os.path.join(out_dir, "pilot.png"))
    with open(os.path.join(out_dir, "pilot.json"), "w") as f:
        json.dump(report, f, indent=1, default=float)

    ssim = report["visible_ssim"]
    print(f"\ntarget {j}: d_norm {report['d_norm']:.3f}, holes {report['hole_fraction']:.1%}, "
          f"visible PSNR {report['visible_psnr']:.2f} dB"
          + (f", SSIM {ssim:.3f}" if ssim is not None else " (pip install scikit-image for SSIM)"))
    print(f"reprojection error at target: {err_aligned:.2f}px aligned vs {err_unscaled:.1f}px unscaled")
    print(f"saved {out_dir}/pilot.png")


if __name__ == "__main__":
    main()
