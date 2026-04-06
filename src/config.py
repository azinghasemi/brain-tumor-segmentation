"""
Central configuration for the brain tumor segmentation pipeline.
All flags and hyperparameters live here — change once, applies everywhere.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = os.environ.get("DATA_DIR", "data/lgg-mri-segmentation")
OUTPUT_DIR  = os.environ.get("OUTPUT_DIR", "outputs")
LOG_DIR     = os.path.join(OUTPUT_DIR, "logs")
CKPT_DIR    = os.path.join(OUTPUT_DIR, "checkpoints")
PRED_DIR    = os.path.join(OUTPUT_DIR, "predictions")

for _d in [OUTPUT_DIR, LOG_DIR, CKPT_DIR, PRED_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── Mode flags ────────────────────────────────────────────────────────────────
DEBUG     = bool(int(os.environ.get("DEBUG", "0")))
FAST_MODE = bool(int(os.environ.get("FAST_MODE", "0")))

# ── Image settings ────────────────────────────────────────────────────────────
IMG_SIZE    = (128, 128) if FAST_MODE else (256, 256)   # (H, W) in pixels
CHANNELS    = 3                                          # RGB input
MASK_THRESH = 127                                        # pixel > thresh → tumor

# ── Training hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE   = 8  if (DEBUG or FAST_MODE) else 16
EPOCHS       = 1  if DEBUG else 30
LEARNING_RATE = 1e-4
RANDOM_STATE  = 42
VAL_SPLIT     = 0.20   # 80/20 train-validation

# ── Dataset limits ────────────────────────────────────────────────────────────
MAX_SAMPLES  = 50   if DEBUG else None   # None = use all
MAX_TRAIN    = 400  if FAST_MODE else None
MAX_VAL      = 100  if FAST_MODE else None

# ── Augmentation ──────────────────────────────────────────────────────────────
AUG_ROTATIONS       = [0, 90, 180, 270]   # degrees
AUG_BRIGHTNESS_RANGE = (0.85, 1.15)       # multiplicative factor
AUG_GAUSSIAN_SIGMA  = 8.0                 # Gaussian noise σ (pixel scale 0-255)
AUG_GAUSSIAN_SCALE  = AUG_GAUSSIAN_SIGMA / 255.0  # normalised σ

# ── Post-processing ───────────────────────────────────────────────────────────
PRED_THRESHOLD   = 0.5    # probability → binary mask
MIN_REGION_PIXELS = 50    # remove connected components smaller than this

# ── Normalisation ─────────────────────────────────────────────────────────────
# "minmax" (÷255) or "zscore" (subtract mean, divide std)
NORM_METHOD = "minmax"

# ── Early stopping ────────────────────────────────────────────────────────────
ES_PATIENCE  = 7
ES_MONITOR   = "val_dice_coef"
ES_MODE      = "max"

# ── Model names ───────────────────────────────────────────────────────────────
MODELS = ["unet", "attention_unet"]
