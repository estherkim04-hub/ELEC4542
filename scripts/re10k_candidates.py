"""Screen RealEstate10K camera files and shortlist clips worth downloading.

    python scripts/re10k_candidates.py --txt-dir data/RealEstate10K/test --sample 300 --top 15

Ranks clips that move a reasonable distance with little rotation, preferring
sideways motion (what our synthetic paths also do). Pick ~5 from the list and
check the URLs still play before running re10k_extract.py.
"""
import argparse
import glob
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dvid.re10k import clip_summary, load_clip  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt-dir", required=True, help="folder of RealEstate10K .txt files (e.g. the test split)")
    ap.add_argument("--sample", type=int, default=300, help="screen a random subset (0 = all)")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--max-rot", type=float, default=15.0)
    ap.add_argument("--min-frames", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.txt_dir, "*.txt")))
    assert files, f"no .txt files in {args.txt_dir}"
    if args.sample and len(files) > args.sample:
        files = random.Random(args.seed).sample(files, args.sample)

    rows = []
    for f in files:
        clip = load_clip(f)
        if len(clip) < args.min_frames:
            continue
        s = clip_summary(clip, args.max_rot)
        s["url"] = clip.url
        rows.append(s)

    # Translation is in dataset units (no metric scale), so only rank within
    # the set; prefer lateral motion, then distance travelled.
    rows.sort(key=lambda r: (r["direction"] != "lateral", -r["max_translation"]))
    print(f"screened {len(files)} clips, {len(rows)} long enough\n")
    print(f"{'clip_id':<20} {'frames':>6} {'dur(s)':>7} {'targets':>8} {'max_t':>7}  direction   url")
    for r in rows[: args.top]:
        print(f"{r['clip_id']:<20} {r['frames']:>6} {r['duration_s']:>7} {r['usable_targets']:>8} "
              f"{r['max_translation']:>7}  {r['direction']:<11} {r['url']}")


if __name__ == "__main__":
    main()
