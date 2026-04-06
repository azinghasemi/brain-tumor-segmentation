"""
Full evaluation of trained U-Net and Attention U-Net models.

Loads best saved weights, runs inference on the entire validation set,
computes per-image metrics, aggregates confusion matrix, and saves:
  - logs/<model_name>_evaluation.txt
  - outputs/predictions/<model_name>_sample_grid.png  (3 validation samples)
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    DATA_DIR, CKPT_DIR, LOG_DIR, PRED_DIR, OUTPUT_DIR,
    IMG_SIZE, CHANNELS, PRED_THRESHOLD,
)
from dataset import load_dataset
from metrics import compute_all
from postprocessing import postprocess, overlay
from models.unet import build_unet
from models.attention_unet import build_attention_unet
from losses import bce_dice_loss, dice_coef


def load_trained_model(name: str, input_shape: tuple) -> tf.keras.Model:
    """Build architecture and load best weights from checkpoint."""
    if name == "UNet":
        model = build_unet(input_shape)
    else:
        model = build_attention_unet(input_shape)

    model.compile(
        optimizer="adam",
        loss=bce_dice_loss,
        metrics=[dice_coef],
    )

    ckpt_path = os.path.join(CKPT_DIR, f"{name}_best.keras")
    if os.path.exists(ckpt_path):
        model.load_weights(ckpt_path)
        print(f"Loaded weights from {ckpt_path}")
    else:
        print(f"WARNING: No checkpoint found at {ckpt_path} — using random weights")

    return model


def measure_inference_time(model: tf.keras.Model,
                            X_val: np.ndarray,
                            n: int = 50) -> float:
    """Mean per-sample inference time in milliseconds."""
    times = []
    for i in range(min(n, len(X_val))):
        x = X_val[i:i+1]
        t0 = time.time()
        model.predict(x, verbose=0)
        times.append((time.time() - t0) * 1000)
    return float(np.mean(times))


def evaluate_model(
    model: tf.keras.Model,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """
    Run post-processed inference on all validation samples.

    Returns
    -------
    dict with keys: dice, iou, precision, recall, TN, FP, FN, TP, report_str
    """
    per_sample = []
    agg_cm = {"TN": 0, "FP": 0, "FN": 0, "TP": 0}

    for i in tqdm(range(len(X_val)), desc=f"Evaluating {model.name}"):
        x = X_val[i:i+1]
        prob = model.predict(x, verbose=0)[0]        # (H, W, 1)
        mask_pred  = postprocess(prob)               # (H, W) binary
        mask_true  = (y_val[i, ..., 0] > 0.5).astype(np.uint8)

        m = compute_all(mask_true, mask_pred, threshold=PRED_THRESHOLD)
        per_sample.append(m)
        for k in agg_cm:
            agg_cm[k] += m[k]

    df = pd.DataFrame(per_sample)
    means = {
        "dice":      df["dice"].mean(),
        "iou":       df["iou"].mean(),
        "precision": df["precision"].mean(),
        "recall":    df["recall"].mean(),
        **agg_cm,
    }

    report = (
        f"=== {model.name} Evaluation Report ===\n"
        f"Samples: {len(X_val)}\n\n"
        f"Dice:      {means['dice']:.4f}\n"
        f"IoU:       {means['iou']:.4f}\n"
        f"Precision: {means['precision']:.4f}\n"
        f"Recall:    {means['recall']:.4f}\n\n"
        f"Pixel-level Confusion Matrix:\n"
        f"  TN={agg_cm['TN']:,}  FP={agg_cm['FP']:,}\n"
        f"  FN={agg_cm['FN']:,}  TP={agg_cm['TP']:,}\n"
    )
    print(report)

    log_path = os.path.join(LOG_DIR, f"{model.name}_evaluation.txt")
    with open(log_path, "w") as f:
        f.write(report)
    print(f"Report saved → {log_path}")

    return {**means, "report": report}


def save_prediction_grid(
    model: tf.keras.Model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_samples: int = 3,
) -> None:
    """
    Save a grid: [MRI | Ground Truth | Prediction | Overlay] × n_samples.
    """
    indices = np.linspace(0, len(X_val) - 1, n_samples, dtype=int)

    fig, axes = plt.subplots(n_samples, 4, figsize=(14, n_samples * 3.5))
    col_titles = ["Input MRI", "Ground Truth", "Prediction", "Overlay"]

    for row, idx in enumerate(indices):
        x = X_val[idx:idx+1]
        prob = model.predict(x, verbose=0)[0]
        pred_mask = postprocess(prob)
        gt_mask   = y_val[idx, ..., 0]
        mri       = X_val[idx]

        ov = overlay(mri, pred_mask)

        axes[row, 0].imshow(mri)
        axes[row, 0].set_ylabel(f"Sample {idx}", fontsize=9)

        axes[row, 1].imshow(gt_mask, cmap="gray")
        axes[row, 2].imshow(pred_mask, cmap="gray")
        axes[row, 3].imshow(ov)

        for ax in axes[row]:
            ax.axis("off")

    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=10, fontweight="bold")

    fig.suptitle(f"{model.name} — Prediction Visualisation", fontsize=12, y=1.01)
    plt.tight_layout()

    path = os.path.join(PRED_DIR, f"{model.name}_prediction_grid.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Prediction grid saved → {path}")


def final_comparison(results: dict) -> None:
    """Print and save the final model comparison table."""
    rows = []
    for name, r in results.items():
        rows.append({
            "Model":           name,
            "Dice":            round(r["dice"],      4),
            "IoU":             round(r["iou"],       4),
            "Precision":       round(r["precision"], 4),
            "Recall":          round(r["recall"],    4),
            "Infer time (ms)": round(r["infer_ms"],  2),
        })
    df = pd.DataFrame(rows)
    print("\n=== FINAL MODEL COMPARISON ===")
    print(df.to_string(index=False))

    path = os.path.join(LOG_DIR, "final_comparison.csv")
    df.to_csv(path, index=False)
    print(f"\nSaved → {path}")


def main():
    input_shape = (*IMG_SIZE, CHANNELS)
    _, _, X_val, y_val, _, _ = load_dataset(DATA_DIR)

    model_names = ["UNet", "AttentionUNet"]
    results = {}

    for name in model_names:
        print(f"\n{'='*60}\n{name}\n{'='*60}")
        model = load_trained_model(name, input_shape)

        infer_ms = measure_inference_time(model, X_val)
        print(f"Mean inference time: {infer_ms:.2f} ms/sample")

        r = evaluate_model(model, X_val, y_val)
        r["infer_ms"] = infer_ms
        results[name] = r

        save_prediction_grid(model, X_val, y_val)

    final_comparison(results)


if __name__ == "__main__":
    main()
