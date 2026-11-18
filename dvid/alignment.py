"""Scale alignment between RealEstate10K poses (no metric scale) and DA3 depth.

We find s such that the dataset translation, scaled by s, is consistent with
the source depth map:

    X_src   = unproject(p_src, depth_src, K)                 (source camera frame)
    X_tgt   = R_rel @ X_src + s * t_rel                      (T_rel = T_tgt @ inv(T_src))
    E(s)    = median_i || project(K, X_tgt_i) - p_tgt_i ||   (pixels)
    s*      = argmin_s E(s)

Correspondences: SIFT + Lowe ratio test + RANSAC on the essential matrix
(using the known K) to drop outliers before the scale search.
After alignment, use  T_aligned = with_scaled_translation(T_rel, s*).
"""
import cv2
import numpy as np

from .geometry import depth_edges, dilate


def match_features(img_a, img_b, K, ratio=0.75, ransac_px=1.0, max_features=8000):
    """SIFT matches between two RGB images, filtered by ratio test + essential-matrix
    RANSAC. Returns (pts_a (N,2), pts_b (N,2)) float64 pixel coordinates."""
    sift = cv2.SIFT_create(nfeatures=max_features)
    ga = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)
    ka, da = sift.detectAndCompute(ga, None)
    kb, db = sift.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return np.empty((0, 2)), np.empty((0, 2))
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < ratio * n.distance]
    if len(good) < 8:
        return np.empty((0, 2)), np.empty((0, 2))
    pa = np.float64([ka[m.queryIdx].pt for m in good])
    pb = np.float64([kb[m.trainIdx].pt for m in good])
    E, inl = cv2.findEssentialMat(pa, pb, K, method=cv2.RANSAC, prob=0.999, threshold=ransac_px)
    if inl is None:
        return np.empty((0, 2)), np.empty((0, 2))
    inl = inl.ravel().astype(bool)
    return pa[inl], pb[inl]


def _lift(pts, depth, K, edge_thresh=0.1):
    """Unproject keypoints using the source depth; drop points without valid,
    reliable depth (invalid or on a depth edge). Returns (X (M,3), keep (N,))."""
    H, W = depth.shape
    u = np.round(pts[:, 0]).astype(int)
    v = np.round(pts[:, 1]).astype(int)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    keep = np.zeros(len(pts), bool)
    bad = dilate(depth_edges(depth, edge_thresh), 1) | (depth <= 0)
    keep[inb] = ~bad[v[inb], u[inb]]
    z = depth[v[keep], u[keep]].astype(np.float64)
    x = (pts[keep, 0] - K[0, 2]) * z / K[0, 0]
    y = (pts[keep, 1] - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], 1), keep


def reprojection_errors(X_src, pts_tgt, K, T_rel, s):
    """Per-point pixel error after moving X_src into the target with scale s."""
    Xt = X_src @ T_rel[:3, :3].T + s * T_rel[:3, 3]
    z = Xt[:, 2]
    ok = z > 1e-6
    err = np.full(len(X_src), np.inf)
    u = K[0, 0] * Xt[ok, 0] / z[ok] + K[0, 2]
    v = K[1, 1] * Xt[ok, 1] / z[ok] + K[1, 2]
    err[ok] = np.hypot(u - pts_tgt[ok, 0], v - pts_tgt[ok, 1])
    return err


def estimate_scale(src_rgb, tgt_rgb, src_depth, K, T_src, T_tgt, s_range=(1e-2, 1e3), n_grid=400):
    """Scale s that maps dataset translation units to the depth map's units.

    Returns dict: scale, median_err_px (at s*), median_err_px_unscaled (at s=1),
    n_matches, n_used, trans_px (how many pixels the translation moves a point at
    median depth: below ~5 px the scale is poorly constrained, use a further target).
    """
    T_rel = T_tgt @ np.linalg.inv(T_src)
    pa, pb = match_features(src_rgb, tgt_rgb, K)
    out = {"n_matches": len(pa), "n_used": 0, "scale": np.nan,
           "median_err_px": np.nan, "median_err_px_unscaled": np.nan, "trans_px": np.nan}
    if len(pa) < 10:
        return out
    X, keep = _lift(pa, src_depth, K)
    pb = pb[keep]
    out["n_used"] = len(X)
    if len(X) < 10:
        return out

    def E(s):
        return float(np.median(reprojection_errors(X, pb, K, T_rel, s)))

    # coarse log-spaced grid, then golden-section refinement around the best
    grid = np.geomspace(*s_range, n_grid)
    errs = np.array([E(s) for s in grid])
    i = int(np.argmin(errs))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, n_grid - 1)]
    g = (np.sqrt(5) - 1) / 2
    a, b = lo + (1 - g) * (hi - lo), lo + g * (hi - lo)
    Ea, Eb = E(a), E(b)
    for _ in range(60):
        if Ea < Eb:
            hi, b, Eb = b, a, Ea
            a = lo + (1 - g) * (hi - lo)
            Ea = E(a)
        else:
            lo, a, Ea = a, b, Eb
            b = lo + g * (hi - lo)
            Eb = E(b)
    s_star = (lo + hi) / 2

    med_z = float(np.median(X[:, 2]))
    out.update(scale=float(s_star), median_err_px=E(s_star), median_err_px_unscaled=E(1.0),
               trans_px=float(K[0, 0] * s_star * np.linalg.norm(T_rel[:3, 3]) / med_z))
    return out


def estimate_clip_scale(results, min_trans_px=5.0, min_used=30):
    """Robust clip-level scale: median over pairs that are well constrained."""
    good = [r["scale"] for r in results
            if np.isfinite(r["scale"]) and r["trans_px"] >= min_trans_px and r["n_used"] >= min_used]
    return (float(np.median(good)) if good else np.nan), len(good)


def with_scaled_translation(T, s):
    """Copy of pose T with its translation multiplied by s."""
    T2 = T.copy()
    T2[:3, 3] *= s
    return T2
