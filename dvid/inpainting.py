"""Member 2: inpainting and compositing.

`inpaint` is a STUB (fills holes with the mean colour of the valid pixels and
slightly perturbs the rest, mimicking a VAE round-trip) so that the pipeline
and conditions B vs C are already distinguishable. Replace it with the real
diffusion inpainter; keep the signature and contract checks.

`selective_composite` is already the final implementation.
"""
import numpy as np

from .contracts import check_mask, check_rgb


def inpaint(warped_rgb, hole_mask, seed=0):
    """Return the inpainting model's RAW full decoded frame (no compositing).

    Must return the model output for ALL pixels, including outside the mask:
    condition B uses it as-is.
    """
    check_rgb(warped_rgb, "warped_rgb")
    check_mask(hole_mask, "hole_mask")

    rng = np.random.default_rng(seed)
    out = warped_rgb.astype(np.int16)
    out += rng.integers(-3, 4, size=out.shape, dtype=np.int16)  # fake VAE drift
    if hole_mask.any() and (~hole_mask).any():
        out[hole_mask] = warped_rgb[~hole_mask].mean(axis=0).astype(np.int16)
    raw = np.clip(out, 0, 255).astype(np.uint8)

    check_rgb(raw, "raw_inpainted_rgb")
    return raw


def selective_composite(warped_rgb, raw_inpainted_rgb, hole_mask):
    """Keep warped pixels where valid, take inpainted pixels only inside holes."""
    check_rgb(warped_rgb, "warped_rgb")
    check_rgb(raw_inpainted_rgb, "raw_inpainted_rgb")
    check_mask(hole_mask, "hole_mask")

    completed = warped_rgb.copy()
    completed[hole_mask] = raw_inpainted_rgb[hole_mask]

    check_rgb(completed, "completed_rgb")
    return completed
