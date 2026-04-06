"""
BCE + Dice combined loss function and Dice coefficient metric.
Both are implemented as TensorFlow/Keras-compatible functions.

Combined loss = Binary Cross-Entropy + (1 - Dice)
  - BCE:  penalises each pixel independently → handles class imbalance
  - Dice: directly optimises the overlap metric → stable with sparse masks
"""

import tensorflow as tf


def dice_coef(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    """
    Per-batch Dice coefficient.

    Parameters
    ----------
    y_true  : ground truth binary mask  (batch, H, W, 1)
    y_pred  : predicted probabilities   (batch, H, W, 1)
    smooth  : Laplace smoothing to avoid 0/0

    Returns
    -------
    Scalar Dice coefficient in [0, 1].
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return 1.0 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Combined BCE + Dice loss.
    Equal weighting (0.5 each) gives stable gradients in both sparse
    and dense tumour regions.
    """
    bce  = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce  = tf.reduce_mean(bce)
    dloss = dice_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * dloss
