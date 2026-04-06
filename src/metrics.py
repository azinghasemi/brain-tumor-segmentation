"""
Per-image segmentation metrics: Dice, IoU, Precision, Recall.
All functions operate on flattened binary arrays (numpy).
Used during evaluation (not training — training uses TF ops from losses.py).
"""

import numpy as np


def _binary(arr: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (arr > threshold).astype(np.uint8).flatten()


def dice_score(y_true: np.ndarray, y_pred: np.ndarray,
               threshold: float = 0.5, smooth: float = 1.0) -> float:
    t = _binary(y_true)
    p = _binary(y_pred, threshold)
    intersection = (t * p).sum()
    return (2.0 * intersection + smooth) / (t.sum() + p.sum() + smooth)


def iou_score(y_true: np.ndarray, y_pred: np.ndarray,
              threshold: float = 0.5, smooth: float = 1.0) -> float:
    t = _binary(y_true)
    p = _binary(y_pred, threshold)
    intersection = (t * p).sum()
    union = t.sum() + p.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def precision_score(y_true: np.ndarray, y_pred: np.ndarray,
                    threshold: float = 0.5, smooth: float = 1.0) -> float:
    t = _binary(y_true)
    p = _binary(y_pred, threshold)
    tp = (t * p).sum()
    fp = ((1 - t) * p).sum()
    return (tp + smooth) / (tp + fp + smooth)


def recall_score(y_true: np.ndarray, y_pred: np.ndarray,
                 threshold: float = 0.5, smooth: float = 1.0) -> float:
    t = _binary(y_true)
    p = _binary(y_pred, threshold)
    tp = (t * p).sum()
    fn = (t * (1 - p)).sum()
    return (tp + smooth) / (tp + fn + smooth)


def confusion_matrix_counts(y_true: np.ndarray, y_pred: np.ndarray,
                             threshold: float = 0.5) -> dict:
    """Return pixel-level TN, FP, FN, TP counts."""
    t = _binary(y_true)
    p = _binary(y_pred, threshold)
    tp = int((t * p).sum())
    fp = int(((1 - t) * p).sum())
    fn = int((t * (1 - p)).sum())
    tn = int(((1 - t) * (1 - p)).sum())
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp}


def compute_all(y_true: np.ndarray, y_pred: np.ndarray,
                threshold: float = 0.5) -> dict:
    """Compute all metrics for a single image pair."""
    return {
        "dice":      dice_score(y_true, y_pred, threshold),
        "iou":       iou_score(y_true, y_pred, threshold),
        "precision": precision_score(y_true, y_pred, threshold),
        "recall":    recall_score(y_true, y_pred, threshold),
        **confusion_matrix_counts(y_true, y_pred, threshold),
    }
