"""Optional depth model for add_to_scene (Condition D). GPU / Colab only.

    from dvid import geometry, depth_models
    geometry.set_depth_estimator(depth_models.da3_estimator())   # once per session

Without this, add_to_scene uses fill_background_depth (no model needed), which
assumes revealed regions are background. The DA3 estimate is re-aligned to the
scene (scale + shift) inside add_to_scene, so relative or metric output both work.
"""
import numpy as np
from PIL import Image

from .contracts import H, W


def da3_estimator(model_id="depth-anything/DA3NESTED-GIANT-LARGE", device=None):
    """Return fn(rgb uint8 (H,W,3)) -> depth float32 (H,W) using Depth Anything 3."""
    import torch
    from depth_anything_3.api import DepthAnything3

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained(model_id).to(device)

    def estimate(rgb):
        pred = model.inference([rgb])
        d = np.asarray(pred.depth[0], dtype=np.float32)
        d = np.array(Image.fromarray(d).resize((W, H), Image.BILINEAR), dtype=np.float32)
        d[~np.isfinite(d)] = 0
        return d

    return estimate
