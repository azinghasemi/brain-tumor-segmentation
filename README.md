# Brain Tumor Segmentation — U-Net vs Attention U-Net

**Binary segmentation of brain tumors in MRI scans using deep learning.**  
Systematic comparison of U-Net and Attention U-Net under limited data conditions.

---

## Results

| Model | Dice | IoU | Precision | Recall | Params | Time/epoch |
|-------|------|-----|-----------|--------|--------|------------|
| **U-Net** | **0.9067** | **0.8788** | 0.9494 | **0.9215** | 31.4M | 9.94 s |
| Attention U-Net | 0.8785 | 0.8523 | **0.9527** | 0.8932 | 31.9M | 10.48 s |

**U-Net outperforms Attention U-Net overall** — higher Dice, IoU, and Recall.  
Attention U-Net shows slightly higher Precision (fewer false positives).  
U-Net is also ~5% faster per epoch with 517K fewer parameters.

---

## Dataset

**LGG MRI Segmentation** — Kaggle ([link](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation))

- 3,929 brain MRI images with paired pixel-level tumor masks
- Binary segmentation: background (pixel ≤ 127) vs tumor (pixel > 127)
- Images stored as `.tif`; masks named `<image>_mask.tif`
- 80/20 train–validation split (random_state=42)

---

## Pipeline Overview

```
Raw MRI + Masks
    │
    ▼
Preprocessing (resize 256×256, MinMax normalize ÷255)
    │
    ▼
Augmentation (rotations 0/90/180/270°, brightness ±15%, Gaussian noise σ=8)
    │
    ├──► U-Net (encoder-decoder, 4 blocks: 64/128/256/512 filters)
    └──► Attention U-Net (same backbone + attention gates on skip connections)
         │
         ▼
    BCE + Dice Loss │ Adam lr=1e-4 │ Early stopping (val Dice)
         │
         ▼
    Post-processing (threshold 0.5 → remove <50px → keep largest region)
         │
         ▼
    Evaluation: Dice, IoU, Precision, Recall, Pixel-level Confusion Matrix
```

---

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Loss function | BCE + Dice | BCE handles pixel-level imbalance; Dice optimises overlap directly |
| Normalisation | MinMax ÷255 | Simple, preserves relative intensity; Z-score available via flag |
| Augmentation | Medical-safe only | Rotations + brightness + noise; no flips that would break anatomy |
| Post-processing | Threshold → morphology | Removes spurious small regions, keeps single largest tumor |
| Image size | 128 (fast) / 256 (full) | Controlled via `FAST_MODE` flag |

---

## Project Structure

```
brain-tumor-segmentation/
├── src/
│   ├── config.py             # All hyperparameters and flags
│   ├── dataset.py            # Data loading, normalisation, train/val split
│   ├── augmentation.py       # Medical-safe augmentation pipeline
│   ├── losses.py             # BCE-Dice combined loss
│   ├── metrics.py            # Dice, IoU, Precision, Recall per-image
│   ├── postprocessing.py     # Threshold → small-region removal → keep largest
│   ├── train.py              # Training loop with early stopping + checkpointing
│   ├── evaluate.py           # Full validation-set evaluation + confusion matrix
│   └── models/
│       ├── unet.py           # Baseline U-Net (31.4M params)
│       └── attention_unet.py # Attention U-Net (31.9M params)
├── notebooks/
│   └── brain_tumor_segmentation.ipynb  # End-to-end Colab-compatible notebook
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
pip install -r requirements.txt

# Place the LGG dataset at data/lgg-mri-segmentation/
# (download from Kaggle, or mount Google Drive in Colab)

# Train both models
python src/train.py

# Evaluate
python src/evaluate.py

# Or run the full pipeline in Colab:
# Open notebooks/brain_tumor_segmentation.ipynb
```

---

## Future Work

- Patient-level train/val split (avoid slice leakage)
- Multimodal MRI (T1, T2, FLAIR fusion)
- Compare normalisation strategies (MinMax vs Z-score)
- Stronger baselines: nnU-Net, DeepLabV3+, Swin-UNet
- 3D volumetric segmentation
