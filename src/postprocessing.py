"""
Post-processing pipeline for raw model output (probability map → clean binary mask).

Steps:
  1. Threshold at 0.5 → binary mask
  2. Remove connected components with fewer than MIN_REGION_PIXELS pixels
  3. Keep only the single largest remaining component

This cleans up spurious predictions and enforces the single-tumour assumption
appropriate for the LGG dataset.
"""

import numpy as np
from skimage import measure, morphology
from config import PRED_THRESHOLD, MIN_REGION_PIXELS


def postprocess(prob_map: np.ndarray,
                threshold: float = PRED_THRESHOLD,
                min_pixels: int = MIN_REGION_PIXELS) -> np.ndarray:
    """
    Convert a probability map to a clean binary mask.

    Parameters
    ----------
    prob_map   : float array (H, W) or (H, W, 1), values in [0, 1]
    threshold  : probability cutoff for tumour class
    min_pixels : minimum connected-component size to retain

    Returns
    -------
    Binary mask (H, W), dtype uint8, values in {0, 1}
    """
    # --- squeeze trailing channel if present ---
    if prob_map.ndim == 3 and prob_map.shape[-1] == 1:
        prob_map = prob_map[..., 0]

    # Step 1: threshold
    binary = (prob_map > threshold).astype(np.uint8)

    if binary.sum() == 0:
        return binary   # no prediction at all

    # Step 2 & 3: label connected components, remove small ones, keep largest
    labeled = measure.label(binary, connectivity=2)
    regions = measure.regionprops(labeled)

    # Remove regions smaller than min_pixels
    regions = [r for r in regions if r.area >= min_pixels]
    if not regions:
        return np.zeros_like(binary)

    # Keep largest
    largest = max(regions, key=lambda r: r.area)
    clean = np.zeros_like(binary)
    clean[labeled == largest.label] = 1

    return clean


def overlay(mri: np.ndarray, mask: np.ndarray,
            color: tuple = (1.0, 0.0, 0.0),
            alpha: float = 0.4) -> np.ndarray:
    """
    Blend a binary mask onto an MRI image for visualisation.

    Parameters
    ----------
    mri   : float array (H, W, 3) normalised to [0, 1]
    mask  : binary array (H, W) or (H, W, 1)
    color : RGB colour for the tumour overlay
    alpha : opacity of the overlay

    Returns
    -------
    Blended float array (H, W, 3)
    """
    if mask.ndim == 3:
        mask = mask[..., 0]

    out = mri.copy()
    tumor_px = mask == 1
    for c, col_val in enumerate(color):
        out[..., c][tumor_px] = (
            (1 - alpha) * out[..., c][tumor_px] + alpha * col_val
        )
    return np.clip(out, 0.0, 1.0)
