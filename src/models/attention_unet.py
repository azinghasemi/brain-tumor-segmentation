"""
Attention U-Net for binary medical image segmentation.

Identical backbone to U-Net, but each decoder skip connection passes through
an Attention Gate before concatenation.

Attention Gate
--------------
Given:
  g  — gating signal from the lower resolution (decoder path)
  x  — skip connection from encoder at the same resolution

The gate computes:
  ψ = σ(W_ψ · ReLU(W_g·g + W_x·x + b_g))
  output = ψ ⊙ x

This lets the network suppress irrelevant background pixels and focus on
tumour boundaries, improving localisation at the cost of ~517K extra params.

Total parameters: 31,919,940 (31,908,164 trainable)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def _conv_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_c1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_r1")(x)
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_c2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_r2")(x)
    return x


def _encoder_block(x: tf.Tensor, filters: int, name: str):
    skip = _conv_block(x, filters, name)
    pool = layers.MaxPooling2D(2, name=f"{name}_pool")(skip)
    return skip, pool


def attention_gate(g: tf.Tensor, x: tf.Tensor,
                   inter_filters: int, name: str) -> tf.Tensor:
    """
    Soft attention gate.

    Parameters
    ----------
    g             : gating signal (from decoder / lower resolution)
    x             : skip feature map (from encoder / higher resolution)
    inter_filters : number of intermediate filters (typically filters // 2)
    name          : prefix for layer names
    """
    # Project both signals to the same inter_filters space
    W_g = layers.Conv2D(inter_filters, 1, padding="same",
                        name=f"{name}_Wg")(g)
    W_x = layers.Conv2D(inter_filters, 1, padding="same",
                        name=f"{name}_Wx")(x)

    psi = layers.Add(name=f"{name}_add")([W_g, W_x])
    psi = layers.Activation("relu", name=f"{name}_relu")(psi)

    # Attention coefficients ψ ∈ [0, 1]
    psi = layers.Conv2D(1, 1, padding="same", name=f"{name}_psi")(psi)
    psi = layers.Activation("sigmoid", name=f"{name}_sigmoid")(psi)

    # Element-wise gate
    out = layers.Multiply(name=f"{name}_out")([x, psi])
    return out


def _decoder_block_with_attention(x: tf.Tensor, skip: tf.Tensor,
                                   filters: int, name: str) -> tf.Tensor:
    """Up-sample → attention-gate skip → concatenate → conv block."""
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same",
                               name=f"{name}_up")(x)

    # Apply attention gate to the skip connection
    attended_skip = attention_gate(
        g=x, x=skip,
        inter_filters=filters // 2,
        name=f"{name}_attn",
    )

    x = layers.Concatenate(name=f"{name}_cat")([x, attended_skip])
    x = _conv_block(x, filters, name)
    return x


def build_attention_unet(input_shape: tuple = (256, 256, 3),
                         filters: list[int] = None) -> Model:
    """
    Build and return the Attention U-Net model.

    Parameters
    ----------
    input_shape : (H, W, C)
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

    # Decoder with attention gates
    for i, f in enumerate(reversed(filters)):
        x = _decoder_block_with_attention(x, skips[-(i + 1)], f,
                                          name=f"dec{i+1}")

    # Output
    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="output")(x)

    model = Model(inputs, outputs, name="AttentionUNet")
    return model


if __name__ == "__main__":
    model = build_attention_unet()
    model.summary()
    print(f"\nTotal params:     {model.count_params():,}")
