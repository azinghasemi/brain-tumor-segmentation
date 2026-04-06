"""
Data loading, normalisation, and train/val splitting.

Expected dataset layout (LGG MRI Segmentation from Kaggle):
    data/lgg-mri-segmentation/
        kaggle_3m/
            TCGA_CS_4941_19960909/
                TCGA_CS_4941_19960909_1.tif
                TCGA_CS_4941_19960909_1_mask.tif
                ...
            ...
        data.csv
"""

import os
import glob
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import (
    DATA_DIR, IMG_SIZE, MASK_THRESH, NORM_METHOD,
    VAL_SPLIT, RANDOM_STATE, MAX_SAMPLES, MAX_TRAIN, MAX_VAL,
    AUG_GAUSSIAN_SCALE, AUG_BRIGHTNESS_RANGE, AUG_ROTATIONS,
)


# ---------------------------------------------------------------------------
# Path discovery
# ---------------------------------------------------------------------------

def discover_pairs(data_dir: str = DATA_DIR) -> list[tuple[str, str]]:
    """
    Return sorted list of (image_path, mask_path) tuples.
    Masks are identified by the '_mask' suffix before the extension.
    """
    pattern  = os.path.join(data_dir, "**", "*.tif")
    all_tifs = glob.glob(pattern, recursive=True)

    pairs = []
    for path in sorted(all_tifs):
        if "_mask" in os.path.basename(path):
            continue
        mask_path = path.replace(".tif", "_mask.tif")
        if os.path.exists(mask_path):
            pairs.append((path, mask_path))

    return pairs


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalise(img: np.ndarray, method: str = NORM_METHOD) -> np.ndarray:
    img = img.astype(np.float32)
    if method == "minmax":
        return img / 255.0
    elif method == "zscore":
        mean, std = img.mean(), img.std()
        return (img - mean) / (std + 1e-8)
    else:
        raise ValueError(f"Unknown normalisation method: {method}")


# ---------------------------------------------------------------------------
# Single image/mask loader
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """Load RGB MRI image, resize to IMG_SIZE, normalise."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)        # bilinear for images
    return normalise(img)


def load_mask(path: str) -> np.ndarray:
    """Load grayscale mask, resize (nearest neighbour), binarise, add channel."""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
    mask = (mask > MASK_THRESH).astype(np.float32)
    return mask[..., np.newaxis]   # shape (H, W, 1)


# ---------------------------------------------------------------------------
# Augmentation (training only)
# ---------------------------------------------------------------------------

def _rotate90(img: np.ndarray, mask: np.ndarray, k: int):
    """Rotate image and mask by k×90°."""
    return np.rot90(img, k), np.rot90(mask, k)


def augment(img: np.ndarray, mask: np.ndarray):
    """
    Apply a randomly chosen augmentation:
      - Rotation (0/90/180/270°)
      - Brightness scaling
      - Gaussian noise
    Each call picks ONE transformation at random.
    Applied to both image and mask identically (except brightness/noise → image only).
    """
    choice = np.random.randint(0, 3)

    if choice == 0:
        k = np.random.choice([1, 2, 3])   # skip 0 (no-op)
        img, mask = _rotate90(img, mask, k)

    elif choice == 1:
        factor = np.random.uniform(*AUG_BRIGHTNESS_RANGE)
        img = np.clip(img * factor, 0.0, 1.0)

    else:
        noise = np.random.normal(0, AUG_GAUSSIAN_SCALE, img.shape).astype(np.float32)
        img = np.clip(img + noise, 0.0, 1.0)

    return img, mask


# ---------------------------------------------------------------------------
# Batch loader
# ---------------------------------------------------------------------------

def load_batch(
    pairs: list[tuple[str, str]],
    apply_augmentation: bool = False,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a list of (image_path, mask_path) pairs into numpy arrays.

    Returns
    -------
    X : float32 array (N, H, W, 3)
    y : float32 array (N, H, W, 1)
    """
    if max_samples is not None:
        pairs = pairs[:max_samples]

    images, masks = [], []
    for img_path, mask_path in tqdm(pairs, desc="Loading", unit="img"):
        img  = load_image(img_path)
        mask = load_mask(mask_path)
        if apply_augmentation:
            img, mask = augment(img, mask)
        images.append(img)
        masks.append(mask)

    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)


# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------

def get_splits(data_dir: str = DATA_DIR):
    """
    Discover all pairs, apply global MAX_SAMPLES limit,
    then split 80/20. Returns train and val pair lists.
    """
    pairs = discover_pairs(data_dir)
    print(f"Found {len(pairs)} image-mask pairs in {data_dir}")

    if MAX_SAMPLES is not None:
        pairs = pairs[:MAX_SAMPLES]
        print(f"  Limited to {len(pairs)} samples (MAX_SAMPLES={MAX_SAMPLES})")

    train_pairs, val_pairs = train_test_split(
        pairs, test_size=VAL_SPLIT, random_state=RANDOM_STATE
    )
    print(f"  Train: {len(train_pairs)} | Val: {len(val_pairs)}")
    return train_pairs, val_pairs


def load_dataset(data_dir: str = DATA_DIR):
    """
    Full load: discover → split → load into memory.

    Returns
    -------
    X_train, y_train, X_val, y_val : numpy arrays
    train_pairs, val_pairs         : path lists (for later reference)
    """
    train_pairs, val_pairs = get_splits(data_dir)

    print("\nLoading training data...")
    X_train, y_train = load_batch(
        train_pairs, apply_augmentation=True,
        max_samples=MAX_TRAIN,
    )
    print("Loading validation data...")
    X_val, y_val = load_batch(
        val_pairs, apply_augmentation=False,
        max_samples=MAX_VAL,
    )

    print(f"\nX_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}   | y_val:   {y_val.shape}")

    return X_train, y_train, X_val, y_val, train_pairs, val_pairs
