# Screenshot Guide

Take these 3 screenshots and save them here. All PNG, 1400px wide minimum.

---

## 01_segmentation_results.png
**What:** A 3-column figure: MRI input | ground truth mask | U-Net prediction
- Open `notebooks/brain_tumor_segmentation.ipynb` in Colab or Jupyter
- Run to the visualization cell (usually at the end of the notebook)
- Pick a scan where the tumor is clearly visible
- Screenshot the 3-panel figure

## 02_training_curves.png
**What:** Validation Dice over epochs — both models on the same chart
- The cell that plots training history
- U-Net line should be above Attention U-Net line
- x-axis = epoch, y-axis = val_dice

## 03_model_comparison.png
**What:** Bar chart comparing all 4 metrics (Dice, IoU, Precision, Recall) side by side
- Usually the last evaluation cell
- U-Net bars in blue, Attention U-Net in orange

---

All three can be captured from the Colab notebook without any local GPU setup.
After adding, commit and push — README already references these filenames.
