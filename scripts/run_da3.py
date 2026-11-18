"""Run Depth Anything 3 on images and save working-resolution RGB, depth and K.

Run this on Colab (GPU). Then copy the output folder to data/ on your laptop.

    !pip install depth-anything-3
    !python scripts/run_da3.py --images data/photos/*.jpg --out /content/drive/MyDrive/elec4542/depth

Per image it writes  <out>/<stem>/rgb.jpg, depth.npy (float32, metres, z-depth),
K.npy (float64, pixels at working res), preview.png.

Models
------
--model nested (default): DA3NESTED-GIANT-LARGE. Outputs metres AND estimates
    intrinsics. Use for your own photos (unknown focal length).
--model metric: DA3METRIC-LARGE (smaller). Outputs *canonical* depth for a
    300 px reference focal length; converted with  depth_m = out * f_px / 300.
    Needs the true focal length: pass --fx (pixels at working resolution),
    e.g. for RealEstate10K frames: fx = fx_normalised * 1024.

Images are centre-cropped to 16:9 and resized to 1024x576 BEFORE inference, so
RGB, depth and K all refer to the same pixels.
"""
import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dvid.contracts import H, W  # noqa: E402

MODEL_IDS = {
    "nested": "depth-anything/DA3NESTED-GIANT-LARGE",
    "metric": "depth-anything/DA3METRIC-LARGE",
}


def load_working_rgb(path):
    """Centre-crop to W:H aspect ratio, resize to (W, H). Returns uint8 RGB."""
    img = Image.open(path).convert("RGB")
    w0, h0 = img.size
    target = W / H
    if w0 / h0 > target:  # too wide -> crop width
        new_w = round(h0 * target)
        left = (w0 - new_w) // 2
        img = img.crop((left, 0, left + new_w, h0))
    else:  # too tall -> crop height
        new_h = round(w0 / target)
        top = (h0 - new_h) // 2
        img = img.crop((0, top, w0, top + new_h))
    return np.asarray(img.resize((W, H), Image.BICUBIC), dtype=np.uint8)


def to_working_res(depth_proc, K_proc):
    """Resize depth from DA3's processing resolution to (H, W); rescale K to match."""
    h, w = depth_proc.shape
    depth = np.asarray(
        Image.fromarray(depth_proc.astype(np.float32)).resize((W, H), Image.BILINEAR),
        dtype=np.float32,
    )
    K = None
    if K_proc is not None:
        K = np.asarray(K_proc, dtype=np.float64).copy()
        K[0, :] *= W / w
        K[1, :] *= H / h
        K[2] = [0, 0, 1]
    return depth, K, (w / W)  # last value: processing-res pixels per working-res pixel


def save_preview(path, rgb, depth):
    d = depth.copy()
    valid = d > 0
    if valid.any():
        lo, hi = np.percentile(d[valid], [2, 98])
        d = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    vis = (255 * (1 - d)).astype(np.uint8)  # near = bright
    vis = np.repeat(vis[..., None], 3, axis=2)
    Image.fromarray(np.concatenate([rgb, vis], axis=1)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", choices=MODEL_IDS, default="nested")
    ap.add_argument("--fx", type=float, default=None,
                    help="focal length in px at working res (required for --model metric)")
    args = ap.parse_args()
    if args.model == "metric" and args.fx is None:
        ap.error("--model metric needs --fx (true focal length in px at 1024x576)")

    import torch
    from depth_anything_3.api import DepthAnything3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_IDS[args.model]} on {device} ...")
    model = DepthAnything3.from_pretrained(MODEL_IDS[args.model]).to(device)

    paths = sorted(p for pattern in args.images for p in glob.glob(pattern))
    assert paths, "no images matched"
    os.makedirs(args.out, exist_ok=True)

    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        rgb = load_working_rgb(path)
        # One image per call = monocular inference (no multi-view mixing).
        pred = model.inference([rgb])
        depth_proc = np.asarray(pred.depth[0], dtype=np.float32)
        K_proc = None if getattr(pred, "intrinsics", None) is None else pred.intrinsics[0]
        depth, K, proc_per_work = to_working_res(depth_proc, K_proc)

        if args.model == "metric":
            f_proc = args.fx * proc_per_work          # focal at DA3's processing res
            depth = depth * f_proc / 300.0            # canonical -> metres
            K = np.array([[args.fx, 0, W / 2], [0, args.fx, H / 2], [0, 0, 1]], dtype=np.float64)
        assert K is not None, "model returned no intrinsics; use --model metric with --fx"

        depth[~np.isfinite(depth)] = 0
        depth[depth < 0] = 0

        od = os.path.join(args.out, stem)
        os.makedirs(od, exist_ok=True)
        Image.fromarray(rgb).save(os.path.join(od, "rgb.jpg"), quality=95)
        np.save(os.path.join(od, "depth.npy"), depth.astype(np.float32))
        np.save(os.path.join(od, "K.npy"), K)
        save_preview(os.path.join(od, "preview.png"), rgb, depth)

        v = depth[depth > 0]
        fov = np.degrees(2 * np.arctan(W / (2 * K[0, 0])))
        print(f"{stem}: depth median {np.median(v):.2f} m (min {v.min():.2f}, max {v.max():.2f}), "
              f"fx {K[0, 0]:.0f}px -> horizontal FOV {fov:.0f} deg")

    print("Sanity check: indoor rooms should give median depth ~2-5 m and FOV ~50-80 deg.")


if __name__ == "__main__":
    main()
