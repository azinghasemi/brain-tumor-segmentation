# Brain Tumor Segmentation — U-Net vs Attention U-Net

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Dice](https://img.shields.io/badge/U--Net_Dice-0.9067-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-LGG_MRI_3929_images-blue)

> **Binary segmentation of brain tumors in MRI scans using deep learning.** Systematic comparison of U-Net and Attention U-Net under limited-data conditions (3,929 images). U-Net achieves Dice = 0.9067.

---

## Results

| Model | Dice | IoU | Precision | Recall | Params | Time/epoch |
|-------|------|-----|-----------|--------|--------|------------|
| **U-Net** | **0.9067** | **0.8788** | 0.9494 | **0.9215** | 31.4M | 9.94 s |
| Attention U-Net | 0.8785 | 0.8523 | **0.9527** | 0.8932 | 31.9M | 10.48 s |

**U-Net outperforms Attention U-Net overall** — higher Dice, IoU, and Recall with 517K fewer parameters and ~5% faster training.

---

## Screenshots

### Segmentation Results — U-Net Predictions

![Segmentation Results](screenshots/01_segmentation_results.png)
*Left: original MRI scan. Middle: ground truth tumor mask. Right: U-Net predicted mask. Dice = 0.9067 — the model captures tumor boundaries with high precision.*

### Training Curves

![Training Curves](screenshots/02_training_curves.png)
*U-Net vs Attention U-Net validation Dice over epochs. U-Net converges faster and plateaus higher. Early stopping (patience=10) prevents overfitting on the small dataset.*

### Model Comparison

![Model Comparison](screenshots/03_model_comparison.png)
*Side-by-side metric comparison: Dice, IoU, Precision, Recall. Attention U-Net gains ~0.03pp on Precision (fewer false positives) but loses on all other metrics.*

---

## Architecture Comparison

```
Input MRI (256×256)
    │
    ▼
Preprocessing: MinMax normalize ÷255 · Augmentation (rotation, brightness, noise)
    │
    ├──► U-Net
    │     Encoder: 4 blocks (64/128/256/512 filters) → Bottleneck → Decoder (skip connections)
    │     Parameters: 31.4M · BCE + Dice Loss · Adam lr=1e-4
    │
    └──► Attention U-Net
          Same backbone + attention gates on skip connections (soft spatial weights)
          Parameters: 31.9M · Same training config
          │
          ▼
    Post-processing: threshold 0.5 → remove <50px regions → keep largest connected region
          │
          ▼
    Metrics: Dice · IoU · Precision · Recall · Pixel-level Confusion Matrix
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

## Dataset

**LGG MRI Segmentation** — Kaggle ([link](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation))
- 3,929 brain MRI images with paired pixel-level tumor masks
- Binary: background (≤ 127) vs tumor (> 127)
- 80/20 train–validation split (random_state=42)

---

## Project Structure

```
brain-tumor-segmentation/
├── src/
│   ├── config.py             ← All hyperparameters and flags
│   ├── dataset.py            ← Data loading, normalisation, train/val split
│   ├── augmentation.py       ← Medical-safe augmentation pipeline
│   ├── losses.py             ← BCE-Dice combined loss
│   ├── metrics.py            ← Dice, IoU, Precision, Recall per-image
│   ├── postprocessing.py     ← Threshold → small-region removal → keep largest
│   ├── train.py              ← Training loop with early stopping + checkpointing
│   ├── evaluate.py           ← Full validation-set evaluation + confusion matrix
│   └── models/
│       ├── unet.py           ← Baseline U-Net (31.4M params)
│       └── attention_unet.py ← Attention U-Net (31.9M params)
├── notebooks/
│   └── brain_tumor_segmentation.ipynb  ← End-to-end Colab-compatible notebook
├── screenshots/
│   └── GUIDE.md
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
pip install -r requirements.txt

# Place LGG dataset at data/lgg-mri-segmentation/
# (download from Kaggle, or mount Google Drive in Colab)

python src/train.py    # trains both models
python src/evaluate.py # metrics + confusion matrix

# Or run end-to-end in Google Colab:
# Open notebooks/brain_tumor_segmentation.ipynb
```

---

## Future Work

- Patient-level train/val split (avoid slice leakage)
- Multimodal MRI (T1, T2, FLAIR fusion)
- Stronger baselines: nnU-Net, Swin-UNet
- 3D volumetric segmentation
