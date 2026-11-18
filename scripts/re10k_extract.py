"""Download one RealEstate10K clip's video and extract the annotated frames.

    python scripts/re10k_extract.py --txt data/RealEstate10K/test/<clip_id>.txt --out data/re10k

Writes  <out>/<clip_id>/
    frames/<timestamp_us>.jpg   centre-cropped to 16:9, resized to 1024x576
    cameras.npz                 timestamps_us (N,), K (N,3,3) at working res,
                                poses (N,4,4) world-to-camera, dataset scale
    video_size.txt              original video width height

Needs: pip install yt-dlp opencv-python   (ffmpeg on PATH helps yt-dlp merge formats)
The downloaded video is deleted afterwards unless --keep-video.
"""
import argparse
import glob
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dvid.contracts import H, W  # noqa: E402
from dvid.re10k import crop_box_16x9, intrinsics_working, load_clip  # noqa: E402


def download(url, out_stem):
    """Download a <=720p video-only stream. Returns the file path or None."""
    cmd = [sys.executable, "-m", "yt_dlp", "-q", "--no-playlist",
           "-f", "bv*[height<=720][ext=mp4]/bv*[height<=720]/b[height<=720]",
           "-o", out_stem + ".%(ext)s", url]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("  download failed:", (r.stderr or r.stdout).strip().splitlines()[-1:])
        return None
    files = glob.glob(out_stem + ".*")
    return files[0] if files else None


def extract_frames(video_path, timestamps_us, out_dir):
    """Read the video sequentially and save the frame nearest to each timestamp."""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"cannot open {video_path}"
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    left, top, cw, ch = crop_box_16x9(vw, vh)
    os.makedirs(out_dir, exist_ok=True)

    targets = sorted(int(t) for t in timestamps_us)
    ti, saved, prev = 0, 0, None   # prev = (time_us, frame)
    half_frame_us = 0.5e6 / fps if fps > 0 else 20000
    while ti < len(targets):
        ok, frame = cap.read()
        if not ok:
            break
        t_us = cap.get(cv2.CAP_PROP_POS_MSEC) * 1000.0
        # save every target that this frame (or the previous one) is closest to
        while ti < len(targets) and targets[ti] <= t_us + half_frame_us:
            best = frame
            if prev is not None and abs(prev[0] - targets[ti]) < abs(t_us - targets[ti]):
                best = prev[1]
            img = cv2.resize(best[top:top + ch, left:left + cw], (W, H), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(out_dir, f"{targets[ti]}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
            ti += 1
        prev = (t_us, frame)
    cap.release()
    return saved, vw, vh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt", required=True)
    ap.add_argument("--out", default="data/re10k")
    ap.add_argument("--keep-video", action="store_true")
    args = ap.parse_args()

    clip = load_clip(args.txt)
    clip_dir = os.path.join(args.out, clip.clip_id)
    os.makedirs(clip_dir, exist_ok=True)
    print(f"{clip.clip_id}: {len(clip)} frames, {clip.duration_s:.1f}s, {clip.url}")

    video = download(clip.url, os.path.join(clip_dir, "video"))
    if video is None:
        print("  -> video unavailable, skip this clip")
        sys.exit(1)

    saved, vw, vh = extract_frames(video, clip.timestamps_us, os.path.join(clip_dir, "frames"))
    print(f"  extracted {saved}/{len(clip)} frames from {vw}x{vh} video")
    if saved < len(clip):
        print("  WARNING: some timestamps were beyond the end of the video")

    K = np.stack([intrinsics_working(k, vw, vh) for k in clip.intrinsics_norm])
    np.savez(os.path.join(clip_dir, "cameras.npz"),
             timestamps_us=clip.timestamps_us, K=K, poses=clip.poses)
    with open(os.path.join(clip_dir, "video_size.txt"), "w") as f:
        f.write(f"{vw} {vh}\n")
    if not args.keep_video:
        os.remove(video)


if __name__ == "__main__":
    main()
