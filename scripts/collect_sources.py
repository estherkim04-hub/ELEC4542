"""Copy each extracted RealEstate10K clip's first (source) frame into one folder,
named <clip_id>.jpg, ready to upload for depth estimation on Colab.

    python scripts/collect_sources.py --re10k data/re10k --out data/re10k_src
"""
import argparse
import glob
import os
import shutil

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re10k", default="data/re10k")
    ap.add_argument("--out", default="data/re10k_src")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    n = 0
    for clip_dir in sorted(glob.glob(os.path.join(args.re10k, "*"))):
        cams = os.path.join(clip_dir, "cameras.npz")
        if not os.path.exists(cams):
            continue
        ts = int(np.load(cams)["timestamps_us"][0])
        src = os.path.join(clip_dir, "frames", f"{ts}.jpg")
        if not os.path.exists(src):
            print(f"skip {os.path.basename(clip_dir)}: source frame {ts}.jpg missing")
            continue
        dst = os.path.join(args.out, os.path.basename(clip_dir) + ".jpg")
        shutil.copy(src, dst)
        print(f"{dst}  (timestamp {ts})")
        n += 1
    print(f"{n} source frames collected in {args.out}")


if __name__ == "__main__":
    main()
