"""Tests for the RealEstate10K loader (step 4-5)."""
import numpy as np

from dvid.contracts import H, W
from dvid.geometry import pose_from_center
from dvid.re10k import (
    candidate_targets, clip_summary, crop_box_16x9, intrinsics_working, load_clip, relative_pose,
)


def write_clip(path, centers, yaw_deg=None, intr=(0.6, 1.07, 0.5, 0.5)):
    lines = ["https://www.youtube.com/watch?v=FAKE"]
    for i, C in enumerate(centers):
        a = np.radians(0 if yaw_deg is None else yaw_deg[i])
        R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])  # cam-to-world
        T = pose_from_center(C, R)
        vals = [33366 * i, *intr, 0, 0, *T[:3, :4].reshape(-1)]
        lines.append(" ".join(f"{v:.9g}" for v in vals))
    path.write_text("\n".join(lines) + "\n")


def test_load_clip_roundtrip(tmp_path):
    centers = [[0.01 * i, 0, 0] for i in range(10)]
    write_clip(tmp_path / "abc123.txt", centers)
    clip = load_clip(tmp_path / "abc123.txt")
    assert clip.clip_id == "abc123" and len(clip) == 10
    assert clip.timestamps_us[1] == 33366
    T_rel = relative_pose(clip.poses[0], clip.poses[9])
    # camera moved +0.09 along x  =>  world-to-camera translation is -0.09
    assert np.allclose(T_rel[:3, 3], [-0.09, 0, 0], atol=1e-9)


def test_intrinsics_16x9_video():
    K = intrinsics_working((0.6, 1.0667, 0.5, 0.5), 1280, 720)
    assert np.allclose(K[0], [0.6 * W, 0, W / 2], atol=1e-6)
    assert np.allclose(K[1, 2], H / 2)


def test_intrinsics_with_crop():
    # 4:3 video gets its top/bottom cropped: cx stays centred, cy recentred
    assert crop_box_16x9(640, 480) == (0, 60, 640, 360)
    K = intrinsics_working((0.5, 0.6667, 0.5, 0.5), 640, 480)
    assert np.isclose(K[0, 2], W / 2) and np.isclose(K[1, 2], H / 2)
    assert np.isclose(K[0, 0], 0.5 * 640 * W / 640)


def test_rotation_filter(tmp_path):
    centers = [[0.02 * i, 0, 0] for i in range(10)]
    yaw = [3.0 * i for i in range(10)]  # 3 deg per frame -> frames 1-5 are <= 15 deg (16 avoids float edge)
    write_clip(tmp_path / "rot.txt", centers, yaw)
    clip = load_clip(tmp_path / "rot.txt")
    idx = [j for j, _, _ in candidate_targets(clip, 0, max_rot_deg=16.0)]
    assert idx == [1, 2, 3, 4, 5]
    assert clip_summary(clip)["direction"] == "lateral"
