"""Member 1: scene construction, rendering, camera paths, scene update.

Conventions (see CONVENTIONS.md): OpenCV camera axes, world-to-camera 4x4
poses, world frame = source camera, z-depth in metres, pixel centres at
integer coordinates.
"""
import numpy as np

from .contracts import H, W, check_depth, check_K, check_mask, check_pose, check_rgb
from .types import Scene

NEAR = 0.05  # metres; points closer than this to the camera are discarded


# --------------------------------------------------------------------------- #
# Small numpy image helpers (no OpenCV dependency)
# --------------------------------------------------------------------------- #
def _shifts(r):
    return [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]


def _shift(a, dy, dx, fill):
    """Return a shifted copy of a 2D array: out[y, x] = a[y - dy, x - dx]."""
    out = np.full_like(a, fill)
    ys = slice(max(dy, 0), H + min(dy, 0))
    yd = slice(max(-dy, 0), H + min(-dy, 0))
    xs = slice(max(dx, 0), W + min(dx, 0))
    xd = slice(max(-dx, 0), W + min(-dx, 0))
    out[ys, xs] = a[yd, xd]
    return out


def dilate(mask, r=1):
    """Binary dilation with a (2r+1)x(2r+1) square."""
    out = mask.copy()
    for dy, dx in _shifts(r):
        out |= _shift(mask, dy, dx, False)
    return out


def depth_edges(depth, rel_thresh=0.1):
    """True where |log depth| jumps by more than rel_thresh to any 4-neighbour.

    These pixels sit on occlusion boundaries; after bilinear upsampling they
    hold in-between depths ("flying pixels") that smear across disocclusions.
    """
    valid = depth > 0
    logd = np.where(valid, np.log(np.maximum(depth, 1e-6)), 0.0)
    edge = np.zeros_like(valid)
    for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        n_logd = _shift(logd, dy, dx, 0.0)
        n_valid = _shift(valid, dy, dx, False)
        edge |= valid & n_valid & (np.abs(logd - n_logd) > rel_thresh)
    return edge


# --------------------------------------------------------------------------- #
# Scene construction
# --------------------------------------------------------------------------- #
def unproject(depth, K, mask=None):
    """Pixels -> 3D points in the camera frame. Returns (points (N,3), v, u)."""
    valid = depth > 0 if mask is None else (depth > 0) & mask
    v, u = np.nonzero(valid)
    z = depth[v, u].astype(np.float64)
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=1), v, u


def scene_from_rgbd(rgb, depth, K, drop_depth_edges=True, edge_thresh=0.1):
    """Lift a source RGB-D image into a Scene (world = source camera frame).

    drop_depth_edges removes pixels on depth discontinuities (and a 1 px band
    around them), so flying pixels don't smear into disoccluded regions. Those
    areas become holes and are inpainted instead.
    """
    check_rgb(rgb)
    check_depth(depth)
    check_K(K)
    keep = None
    if drop_depth_edges:
        keep = ~dilate(depth_edges(depth, edge_thresh), 1)
    pts, v, u = unproject(depth, K, keep)
    return Scene(points=pts.astype(np.float32), colors=rgb[v, u].copy(), K=K)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _fill_cracks(rgb, depth, valid, min_neighbours=5, max_rel_spread=0.05):
    """Fill isolated empty pixels that are surrounded by a consistent surface.

    A pixel is a crack (not a disocclusion) if >= min_neighbours of its 8
    neighbours are valid and those neighbours' depths agree within
    max_rel_spread. Filled with the mean colour / depth of valid neighbours.
    """
    cnt = np.zeros((H, W), np.int32)
    csum = np.zeros((H, W, 3), np.float64)
    dsum = np.zeros((H, W), np.float64)
    dmin = np.full((H, W), np.inf)
    dmax = np.zeros((H, W))
    for dy, dx in _shifts(1):
        if dy == 0 and dx == 0:
            continue
        nv = _shift(valid, dy, dx, False)
        nd = _shift(depth, dy, dx, 0.0)
        nc = np.stack([_shift(rgb[..., c], dy, dx, 0) for c in range(3)], axis=-1)
        cnt += nv
        csum += nc * nv[..., None]
        dsum += nd * nv
        dmin = np.where(nv, np.minimum(dmin, nd), dmin)
        dmax = np.where(nv, np.maximum(dmax, nd), dmax)
    crack = (~valid) & (cnt >= min_neighbours)
    crack &= (dmax - dmin) <= max_rel_spread * np.where(np.isfinite(dmin), dmin, 0)
    n = np.maximum(cnt, 1)
    rgb = rgb.copy()
    depth = depth.copy()
    rgb[crack] = np.round(csum[crack] / n[crack, None]).astype(np.uint8)
    depth[crack] = (dsum[crack] / n[crack]).astype(np.float32)
    return rgb, depth, valid | crack


def render(scene, T, hole_dilate=1, crack_passes=2):
    """Render `scene` from world-to-camera pose `T` (intrinsics = scene.K).

    Forward z-buffer splatting with a 2x2 footprint per point (closes most
    cracks from mild magnification), then crack filling, then the remaining
    empty pixels are holes (disocclusions), dilated by `hole_dilate` px to
    swallow halo pixels at their borders.

    Returns: warped_rgb (H,W,3) uint8, hole_mask (H,W) bool, depth_map (H,W) float32.
    """
    assert isinstance(scene, Scene)
    check_pose(T)
    K = scene.K

    # world -> camera
    P = scene.points.astype(np.float64) @ T[:3, :3].T + T[:3, 3]
    front = P[:, 2] > NEAR
    P, col = P[front], scene.colors[front]
    z = P[:, 2]
    u = K[0, 0] * P[:, 0] / z + K[0, 2]
    v = K[1, 1] * P[:, 1] / z + K[1, 2]

    # 2x2 footprint: each point competes for the 4 pixels around it. A tiny
    # depth penalty by distance-to-pixel-centre makes the best-centred point
    # win among points on the same surface, without letting background beat
    # foreground.
    u0, v0 = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    keys, idxs, src = [], [], []
    for du in (0, 1):
        for dv in (0, 1):
            uu, vv = u0 + du, v0 + dv
            inb = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            dist = np.hypot(uu - u, vv - v)[inb]
            keys.append(z[inb] * (1.0 + 1e-3 * dist))
            idxs.append(vv[inb] * W + uu[inb])
            src.append(np.nonzero(inb)[0])
    key = np.concatenate(keys)
    idx = np.concatenate(idxs)
    src = np.concatenate(src)

    zbuf = np.full(H * W, np.inf)
    np.minimum.at(zbuf, idx, key)
    win = key <= zbuf[idx]
    rgb = np.zeros((H * W, 3), np.uint8)
    depth = np.zeros(H * W, np.float32)
    rgb[idx[win]] = col[src[win]]
    depth[idx[win]] = z[src[win]]
    rgb = rgb.reshape(H, W, 3)
    depth = depth.reshape(H, W)
    valid = np.isfinite(zbuf).reshape(H, W)

    for _ in range(crack_passes):
        rgb, depth, valid = _fill_cracks(rgb, depth, valid)

    hole_mask = ~valid
    if hole_dilate > 0 and hole_mask.any():
        hole_mask = dilate(hole_mask, hole_dilate)
    rgb[hole_mask] = 0
    depth[hole_mask] = 0.0

    check_rgb(rgb, "warped_rgb")
    check_mask(hole_mask, "hole_mask")
    check_depth(depth, "depth_map")
    return rgb, hole_mask, depth


# --------------------------------------------------------------------------- #
# Camera poses and paths
# --------------------------------------------------------------------------- #
def pose_from_center(C, R=None):
    """World-to-camera pose for a camera centred at C (world coords, metres)
    with camera-to-world rotation R (default: same orientation as source)."""
    R = np.eye(3) if R is None else np.asarray(R, np.float64)
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ np.asarray(C, np.float64)
    return T


def camera_center(T):
    """Camera centre in world coordinates for a world-to-camera pose."""
    return -T[:3, :3].T @ T[:3, 3]


def median_depth(depth):
    v = depth[depth > 0]
    return float(np.median(v)) if v.size else 0.0


def camera_path(kind, scene_median_depth, d_norm, n_frames=49):
    """Smooth camera trajectory from the source view.

    kind: 'lateral' (move right), 'dolly' (move forward), 'orbit' (arc around
    the point at median depth, looking at it).
    d_norm: final displacement as a fraction of scene median depth
    (the RQ1 displacement measure). n_frames: use 4n+1 (Wan2.2-compatible).
    Returns a list of 4x4 world-to-camera poses; the first is the identity.
    """
    D = scene_median_depth * d_norm
    s = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n_frames))  # ease in/out, 0 -> 1
    poses = []
    for a in s:
        if kind == "lateral":
            T = pose_from_center([a * D, 0, 0])
        elif kind == "dolly":
            T = pose_from_center([0, 0, a * D])
        elif kind == "orbit":
            r = scene_median_depth
            theta = a * D / r  # arc length D on a circle of radius r
            C = np.array([r * np.sin(theta), 0, r - r * np.cos(theta)])
            fwd = np.array([0, 0, r]) - C
            fwd /= np.linalg.norm(fwd)
            right = np.cross([0, 1, 0], fwd)  # OpenCV: y down
            right /= np.linalg.norm(right)
            down = np.cross(fwd, right)
            T = pose_from_center(C, np.stack([right, down, fwd], axis=1))
        else:
            raise ValueError(f"unknown path kind {kind!r}")
        poses.append(T)
    return poses


def rotation_angle_deg(T_a, T_b):
    """Relative rotation angle between two poses, in degrees."""
    R = T_b[:3, :3] @ T_a[:3, :3].T
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def normalized_displacement(T_src, T_tgt, src_depth):
    """||C_t - C_0|| / median(D_0)  (both must be in the same scale)."""
    return float(np.linalg.norm(camera_center(T_tgt) - camera_center(T_src)) / median_depth(src_depth))


# --------------------------------------------------------------------------- #
# Scene update (Condition D): temporal reuse of inpainted content
# --------------------------------------------------------------------------- #
_DEPTH_ESTIMATOR = None


def set_depth_estimator(fn):
    """Set the monocular depth model used by add_to_scene (or None for 'fill' mode).

    fn(rgb uint8 (H,W,3)) -> depth float (H,W); any scale/shift, since it is
    re-aligned to the scene. On Colab: set_depth_estimator(depth_models.da3_estimator()).
    """
    global _DEPTH_ESTIMATOR
    _DEPTH_ESTIMATOR = fn


def _propagate(depth, valid, axis, reverse):
    """Depth of the nearest valid pixel before each pixel along an axis (0 = none)."""
    d, v = depth, valid
    if axis == 0:
        d, v = d.T, v.T
    if reverse:
        d, v = d[:, ::-1], v[:, ::-1]
    n = d.shape[1]
    idx = np.where(v, np.arange(n)[None, :], -1)
    idx = np.maximum.accumulate(idx, axis=1)
    out = np.where(idx >= 0, np.take_along_axis(d, np.maximum(idx, 0), axis=1), 0.0)
    if reverse:
        out = out[:, ::-1]
    return out.T if axis == 0 else out


def fill_background_depth(depth, holes):
    """Depth for hole pixels from their surroundings, assuming they show BACKGROUND
    (what a moving camera reveals behind an occluder): along each row take the
    farther of the nearest valid depths to the left and right; rows with no valid
    pixel on either side fall back to the farther of up/down."""
    valid = (depth > 0) & ~holes
    left, right = _propagate(depth, valid, 1, False), _propagate(depth, valid, 1, True)
    up, down = _propagate(depth, valid, 0, False), _propagate(depth, valid, 0, True)
    horiz = np.maximum(left, right)
    out = np.where(horiz > 0, horiz, np.maximum(up, down))
    return np.where(holes, out, depth).astype(np.float32)


def _align_to_ring(est, depth, ring, trim=0.2, iters=2):
    """Fit depth ~= a * est + b on ring pixels (least squares, dropping the worst
    `trim` fraction each pass). Returns aligned estimate or None if unreliable."""
    m = ring & (est > 0) & np.isfinite(est)
    if m.sum() < 50:
        return None
    x, y = est[m].astype(np.float64), depth[m].astype(np.float64)
    keep = np.ones(len(x), bool)
    for _ in range(iters + 1):
        A = np.stack([x[keep], np.ones(keep.sum())], 1)
        (a, b), *_ = np.linalg.lstsq(A, y[keep], rcond=None)
        r = np.abs(a * x + b - y)
        keep = r <= np.quantile(r, 1 - trim)
    if a <= 0:
        return None
    return (a * est + b).astype(np.float32)


def add_to_scene(scene, completed_rgb, hole_mask, T, rendered_depth=None,
                 ring_px=4, edge_thresh=0.1, return_stats=False):
    """Add the inpainted pixels (hole_mask == True) of `completed_rgb`, seen from
    pose `T`, to the scene, so later frames re-render them instead of
    re-inpainting (Condition D). Returns a NEW Scene; never mutates `scene`.

    1. Depth for the hole pixels: the depth model set with set_depth_estimator,
       aligned (scale + shift) to the rendered depth in a ring around the holes;
       without a model, fill_background_depth (holes = revealed background).
    2. Reliability: drop points outside [0.8 x nearest ring depth, 1.25 x farthest
       ring depth], closer than NEAR, or on depth edges of the new depth map.
    3. Unproject with scene.K, move to world coordinates, append.

    rendered_depth: render()'s depth for this pose (saves a re-render).
    return_stats=True also returns a dict (mode, added, rejected, a/b fit).
    """
    assert isinstance(scene, Scene)
    check_rgb(completed_rgb, "completed_rgb")
    check_mask(hole_mask, "hole_mask")
    check_pose(T)
    stats = {"mode": "none", "holes": int(hole_mask.sum()), "added": 0, "rejected": 0}
    copy = Scene(points=scene.points.copy(), colors=scene.colors.copy(), K=scene.K)
    if not hole_mask.any():
        return (copy, stats) if return_stats else copy
    if rendered_depth is None:
        _, _, rendered_depth = render(scene, T, hole_dilate=0)
    check_depth(rendered_depth, "rendered_depth")

    ring = dilate(hole_mask, ring_px) & ~hole_mask & (rendered_depth > 0)
    fill = fill_background_depth(rendered_depth, hole_mask)
    new_depth, stats["mode"] = fill, "fill"
    if _DEPTH_ESTIMATOR is not None:
        est = np.asarray(_DEPTH_ESTIMATOR(completed_rgb), dtype=np.float32)
        assert est.shape == (H, W), f"depth estimator returned {est.shape}, expected {(H, W)}"
        aligned = _align_to_ring(est, rendered_depth, ring)
        if aligned is not None:
            new_depth = np.where(hole_mask, aligned, rendered_depth).astype(np.float32)
            stats["mode"] = "estimator"

    keep = hole_mask & (new_depth > NEAR) & np.isfinite(new_depth)
    if ring.any():
        rd = rendered_depth[ring]
        keep &= (new_depth >= 0.8 * rd.min()) & (new_depth <= 1.25 * rd.max())
    keep &= ~(dilate(depth_edges(np.where(keep | ~hole_mask, new_depth, 0).astype(np.float32),
                                 edge_thresh), 1) & hole_mask)
    stats["added"] = int(keep.sum())
    stats["rejected"] = int(hole_mask.sum() - keep.sum())
    if not keep.any():
        return (copy, stats) if return_stats else copy

    P_cam, v, u = unproject(np.where(keep, new_depth, 0).astype(np.float32), scene.K)
    R, t = T[:3, :3], T[:3, 3]
    P_world = (P_cam - t) @ R  # x_w = R^T (x_c - t)
    out = Scene(points=np.concatenate([scene.points, P_world.astype(np.float32)]),
                colors=np.concatenate([scene.colors, completed_rgb[v, u]]), K=scene.K)
    return (out, stats) if return_stats else out
