"""RealEstate10K camera files: parsing, intrinsics/pose conversion, clip screening.

File format (one .txt per clip):
    line 1:  YouTube URL
    line 2+: timestamp_us fx fy cx cy k1 k2 p00 p01 p02 p03 p10 ... p23
             - timestamp in MICROSECONDS from the start of the video
             - fx fy cx cy as FRACTIONS of image width/height
             - k1 k2 unused (always 0)
             - 12 numbers: 3x4 WORLD-TO-CAMERA matrix, row-major
The poses' translation has no metric scale (SLAM); see dvid.alignment.
"""
from dataclasses import dataclass

import numpy as np

from .contracts import H, W
from .geometry import camera_center, rotation_angle_deg


@dataclass
class Clip:
    clip_id: str
    url: str
    timestamps_us: np.ndarray   # (N,) int64
    intrinsics_norm: np.ndarray  # (N, 4) fx fy cx cy as fractions
    poses: np.ndarray           # (N, 4, 4) float64 world-to-camera (dataset scale)

    def __len__(self):
        return len(self.timestamps_us)

    @property
    def duration_s(self):
        return (self.timestamps_us[-1] - self.timestamps_us[0]) / 1e6


def load_clip(path):
    """Parse one RealEstate10K .txt file."""
    import os
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    url = lines[0]
    rows = np.array([[float(x) for x in ln.split()] for ln in lines[1:]], dtype=np.float64)
    assert rows.shape[1] == 19, f"{path}: expected 19 columns, got {rows.shape[1]}"
    poses = np.tile(np.eye(4), (len(rows), 1, 1))
    poses[:, :3, :4] = rows[:, 7:19].reshape(-1, 3, 4)
    return Clip(
        clip_id=os.path.splitext(os.path.basename(path))[0],
        url=url,
        timestamps_us=rows[:, 0].astype(np.int64),
        intrinsics_norm=rows[:, 1:5],
        poses=poses,
    )


def crop_box_16x9(w0, h0):
    """Centre crop (left, top, width, height) of a w0 x h0 frame to the W:H aspect."""
    target = W / H
    if w0 / h0 > target:
        cw, ch = round(h0 * target), h0
    else:
        cw, ch = w0, round(w0 / target)
    return (w0 - cw) // 2, (h0 - ch) // 2, cw, ch


def intrinsics_working(intr_norm, video_w, video_h):
    """Normalised (fx, fy, cx, cy) on the original video -> 3x3 K in pixels at the
    working resolution, accounting for the same centre crop used on the frames."""
    fx, fy, cx, cy = intr_norm
    left, top, cw, ch = crop_box_16x9(video_w, video_h)
    sx, sy = W / cw, H / ch
    return np.array([
        [fx * video_w * sx, 0, (cx * video_w - left) * sx],
        [0, fy * video_h * sy, (cy * video_h - top) * sy],
        [0, 0, 1],
    ], dtype=np.float64)


def relative_pose(T_src, T_tgt):
    """Pose of the target camera expressed in the source camera frame
    (i.e. world-to-camera with world = source camera)."""
    return T_tgt @ np.linalg.inv(T_src)


def candidate_targets(clip, src_idx=0, max_rot_deg=15.0, min_trans=1e-6):
    """Indices of frames after src_idx whose rotation from the source is below
    max_rot_deg, with their rotation angle and (dataset-scale) translation."""
    out = []
    Cs = camera_center(clip.poses[src_idx])
    for j in range(src_idx + 1, len(clip)):
        rot = rotation_angle_deg(clip.poses[src_idx], clip.poses[j])
        trans = float(np.linalg.norm(camera_center(clip.poses[j]) - Cs))
        if rot <= max_rot_deg and trans > min_trans:
            out.append((j, rot, trans))
    return out


def clip_summary(clip, max_rot_deg=15.0):
    """One-line screening info: how far the camera travels (in dataset units)
    before rotating more than max_rot_deg from frame 0."""
    cands = candidate_targets(clip, 0, max_rot_deg)
    reach = max((t for _, _, t in cands), default=0.0)
    usable = len(cands)
    # direction of travel in the source camera frame (x = sideways, z = forward)
    if cands:
        j = max(cands, key=lambda c: c[2])[0]
        d = camera_center(relative_pose(clip.poses[0], clip.poses[j]))
        d = d / (np.linalg.norm(d) + 1e-12)
        direction = "lateral" if abs(d[0]) > abs(d[2]) else "forward/back"
    else:
        direction = "-"
    return {"clip_id": clip.clip_id, "frames": len(clip), "duration_s": round(clip.duration_s, 1),
            "usable_targets": usable, "max_translation": round(reach, 3), "direction": direction}


def select_sequence(poses, src_idx, scale, src_depth, d_band, n_frames=25, max_rot_deg=15.0):
    """Pick frames along the real trajectory for one displacement band.

    Walks forward from src_idx until the aligned normalised displacement first
    reaches d_band (stopping early if rotation from the source exceeds
    max_rot_deg), then samples n_frames indices evenly from src_idx to that
    frame. Returns (indices, d_norms) or None if the band is never reached.
    """
    from .geometry import median_depth

    md = median_depth(src_depth)
    Cs = camera_center(poses[src_idx])
    end = None
    for j in range(src_idx + 1, len(poses)):
        if rotation_angle_deg(poses[src_idx], poses[j]) > max_rot_deg:
            break
        d = scale * float(np.linalg.norm(camera_center(poses[j]) - Cs)) / md
        if d >= d_band:
            end = j
            break
    if end is None or end - src_idx + 1 < n_frames:
        return None
    idx = np.unique(np.round(np.linspace(src_idx, end, n_frames)).astype(int))
    if len(idx) < n_frames:
        return None
    d_norms = [scale * float(np.linalg.norm(camera_center(poses[j]) - Cs)) / md for j in idx]
    return idx, d_norms
