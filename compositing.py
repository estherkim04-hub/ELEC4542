
import numpy as np


def selective_composite(
    warped_rgb,
    raw_inpainted_rgb,
    hole_mask
):
    """
    Replace only pixels inside the hole mask.

    True  = inpainted region
    False = preserve original pixels
    """

    original = np.asarray(warped_rgb)
    generated = np.asarray(raw_inpainted_rgb)
    mask = np.asarray(hole_mask, dtype=bool)

    assert original.shape == generated.shape
    assert mask.shape == original.shape[:2]

    completed = original.copy()
    completed[mask] = generated[mask]

    return completed
