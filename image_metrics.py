
import numpy as np


def masked_mae(reference, prediction, valid_mask):
    ref = np.asarray(reference).astype(np.float64)
    pred = np.asarray(prediction).astype(np.float64)
    mask = np.asarray(valid_mask, dtype=bool)

    assert ref.shape == pred.shape
    assert mask.shape == ref.shape[:2]
    assert mask.any(), "No valid pixels in mask"

    error = np.abs(ref - pred).mean(axis=2)
    return float(error[mask].mean())


def masked_psnr(reference, prediction, valid_mask):
    ref = np.asarray(reference).astype(np.float64)
    pred = np.asarray(prediction).astype(np.float64)
    mask = np.asarray(valid_mask, dtype=bool)

    assert ref.shape == pred.shape
    assert mask.shape == ref.shape[:2]
    assert mask.any(), "No valid pixels in mask"

    mse_map = ((ref - pred) ** 2).mean(axis=2)
    mse = mse_map[mask].mean()

    if mse == 0:
        return float("inf")

    return float(10 * np.log10(255.0 ** 2 / mse))
