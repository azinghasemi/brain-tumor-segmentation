"""
Baseline U-Net for binary medical image segmentation.

Architecture
------------
Encoder: 4 blocks with filter sizes [64, 128, 256, 512]
  Each block: Conv2D(3×3) → BatchNorm → ReLU → Conv2D(3×3) → BatchNorm → ReLU → MaxPool2D

Bottleneck: 1024 filters, no pooling

Decoder: 4 up-blocks (transpose conv + concatenate skip → 2×Conv blocks)

Output: Conv2D(1×1, sigmoid) → binary probability map

Total parameters: 31,402,497 (31,390,721 trainable)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def _conv_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    """Two Conv2D + BatchNorm + ReLU layers."""
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_c1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_r1")(x)

    x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_c2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_r2")(x)
    return x


def _encoder_block(x: tf.Tensor, filters: int, name: str):
    """Conv block followed by MaxPool; returns (skip, pooled)."""
    skip = _conv_block(x, filters, name)
    pool = layers.MaxPooling2D(2, name=f"{name}_pool")(skip)
    return skip, pool


def _decoder_block(x: tf.Tensor, skip: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    """UpSampling (transpose conv) → concatenate skip → conv block."""
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same",
                               name=f"{name}_up")(x)
    x = layers.Concatenate(name=f"{name}_cat")([x, skip])
    x = _conv_block(x, filters, name)
    return x


def build_unet(input_shape: tuple = (256, 256, 3),
               filters: list[int] = None) -> Model:
    """
    Build and return the U-Net model.

    Parameters
    ----------
    input_shape : (H, W, C) — e.g. (256, 256, 3) or (128, 128, 3)
    filters     : encoder filter sizes (default [64, 128, 256, 512])
    """
    if filters is None:
        filters = [64, 128, 256, 512]

    inputs = layers.Input(shape=input_shape, name="input")
    skips  = []
    x      = inputs

    # Encoder
    for i, f in enumerate(filters):
        skip, x = _encoder_block(x, f, name=f"enc{i+1}")
        skips.append(skip)

    # Bottleneck
    x = _conv_block(x, filters[-1] * 2, name="bottleneck")

    # Decoder
    for i, f in enumerate(reversed(filters)):
        x = _decoder_block(x, skips[-(i + 1)], f, name=f"dec{i+1}")

    # Output
    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="output")(x)

    model = Model(inputs, outputs, name="UNet")
    return model


if __name__ == "__main__":
    model = build_unet()
    model.summary()
    print(f"\nTotal params:     {model.count_params():,}")
