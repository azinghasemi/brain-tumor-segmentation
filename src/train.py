"""
Training script — trains U-Net and Attention U-Net, saves best weights.

Usage
-----
    python src/train.py

Environment flags (set before running):
    FAST_MODE=1   → 128×128 images, 400 training samples
    DEBUG=1       → 1 epoch, 50 samples (quick smoke test)
"""

import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    DATA_DIR, CKPT_DIR, OUTPUT_DIR, LOG_DIR,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, IMG_SIZE, CHANNELS,
    ES_PATIENCE, ES_MONITOR, ES_MODE,
)
from dataset import load_dataset
from losses import bce_dice_loss, dice_coef
from models.unet import build_unet
from models.attention_unet import build_attention_unet


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_one_model(
    model: tf.keras.Model,
    X_train, y_train,
    X_val, y_val,
) -> dict:
    """
    Compile, train with early stopping, restore best weights, save to disk.

    Returns
    -------
    dict with keys: history, val_dice_final, val_iou_final, time_per_epoch
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss=bce_dice_loss,
        metrics=[dice_coef],
    )

    ckpt_path = os.path.join(CKPT_DIR, f"{model.name}_best.keras")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=ES_MONITOR,
            patience=ES_PATIENCE,
            mode=ES_MODE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor=ES_MONITOR,
            save_best_only=True,
            mode=ES_MODE,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=ES_MONITOR,
            factor=0.5,
            patience=3,
            mode=ES_MODE,
            verbose=1,
            min_lr=1e-6,
        ),
    ]

    t_start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t_start

    n_epochs       = len(history.history["loss"])
    time_per_epoch = elapsed / max(n_epochs, 1)
    val_dice_final = history.history[ES_MONITOR][-1]

    print(f"\n{model.name} — {n_epochs} epochs | "
          f"{time_per_epoch:.2f} s/epoch | "
          f"final val Dice = {val_dice_final:.4f}")

    return {
        "history":        history.history,
        "val_dice_final": val_dice_final,
        "n_epochs":       n_epochs,
        "time_per_epoch": time_per_epoch,
        "ckpt_path":      ckpt_path,
    }


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_training_curves(results: dict, save_dir: str = OUTPUT_DIR) -> None:
    """Plot loss and Dice curves for both models side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#2c3e50", "#e74c3c"]

    for (name, res), col in zip(results.items(), colors):
        h = res["history"]
        axes[0].plot(h["loss"],     label=f"{name} train", color=col, ls="-")
        axes[0].plot(h["val_loss"], label=f"{name} val",   color=col, ls="--")
        axes[1].plot(h[ES_MONITOR], label=name, color=col)

    axes[0].set_title("Training vs Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE+Dice Loss")
    axes[0].legend(fontsize=9)

    axes[1].set_title("Validation Dice Coefficient")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved → {path}")


def save_comparison_csv(results: dict, save_dir: str = LOG_DIR) -> None:
    """Save a parameter + performance comparison table."""
    rows = []
    for name, res in results.items():
        rows.append({
            "Model":            name,
            "Val Dice (final)": round(res["val_dice_final"], 4),
            "Epochs run":       res["n_epochs"],
            "Time/epoch (s)":   round(res["time_per_epoch"], 2),
        })
    df = pd.DataFrame(rows)
    path = os.path.join(save_dir, "model_comparison.csv")
    df.to_csv(path, index=False)
    print(f"Comparison saved → {path}")
    print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    input_shape = (*IMG_SIZE, CHANNELS)
    print(f"Input shape: {input_shape} | Batch: {BATCH_SIZE} | Max epochs: {EPOCHS}")

    # Build models
    models = {
        "UNet":          build_unet(input_shape),
        "AttentionUNet": build_attention_unet(input_shape),
    }
    for name, m in models.items():
        n = m.count_params()
        print(f"{name}: {n:,} total params")

    # Load data
    X_train, y_train, X_val, y_val, _, _ = load_dataset(DATA_DIR)

    # Train
    results = {}
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {name}")
        print("="*60)
        results[name] = train_one_model(model, X_train, y_train, X_val, y_val)

    # Post-training outputs
    plot_training_curves(results)
    save_comparison_csv(results)


if __name__ == "__main__":
    main()
