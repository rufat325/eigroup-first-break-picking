"""U-Net architecture and training losses for first-break segmentation."""
import tensorflow as tf
from tensorflow.keras import layers, Model

from .config import (
    PATCH_WIDTH, PATCH_HEIGHT, BASE_FILTERS,
    BCE_POS_WEIGHT, DICE_WEIGHT,
)


def _conv_block(x, filters, name):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                      kernel_initializer="he_normal", name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.ReLU(name=f"{name}_relu1")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                      kernel_initializer="he_normal", name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.ReLU(name=f"{name}_relu2")(x)
    return x


def build_unet(input_shape=(PATCH_WIDTH, PATCH_HEIGHT, 1), base=BASE_FILTERS):
    """4-level U-Net with sigmoid output. ~7.8M parameters at base=32."""
    inp = layers.Input(shape=input_shape, name="input")

    c1 = _conv_block(inp, base,     "enc1"); p1 = layers.MaxPool2D(2)(c1)
    c2 = _conv_block(p1,  base*2,   "enc2"); p2 = layers.MaxPool2D(2)(c2)
    c3 = _conv_block(p2,  base*4,   "enc3"); p3 = layers.MaxPool2D(2)(c3)
    c4 = _conv_block(p3,  base*8,   "enc4"); p4 = layers.MaxPool2D(2)(c4)

    bn = _conv_block(p4, base*16, "bottleneck")

    u4 = layers.Conv2DTranspose(base*8, 2, strides=2, padding="same")(bn)
    d4 = _conv_block(layers.Concatenate()([u4, c4]), base*8, "dec4")
    u3 = layers.Conv2DTranspose(base*4, 2, strides=2, padding="same")(d4)
    d3 = _conv_block(layers.Concatenate()([u3, c3]), base*4, "dec3")
    u2 = layers.Conv2DTranspose(base*2, 2, strides=2, padding="same")(d3)
    d2 = _conv_block(layers.Concatenate()([u2, c2]), base*2, "dec2")
    u1 = layers.Conv2DTranspose(base,   2, strides=2, padding="same")(d2)
    d1 = _conv_block(layers.Concatenate()([u1, c1]), base, "dec1")

    out = layers.Conv2D(1, 1, activation="sigmoid", name="output")(d1)
    return Model(inp, out, name="unet")


def weighted_bce(y_true, y_pred, pos_weight=BCE_POS_WEIGHT):
    """BCE with positive-class weighting to fight severe class imbalance.

    First-break pixels are ~0.5% of the target mask; without up-weighting
    the model would just predict zero everywhere.
    """
    eps = 1e-7
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    return tf.reduce_mean(
        -(pos_weight * y_true * tf.math.log(y_pred)
          + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    )


def dice_loss(y_true, y_pred):
    """Soft Dice loss (1 - Dice coefficient), complementing BCE at low recall."""
    eps = 1.0
    yt = tf.reshape(y_true, (tf.shape(y_true)[0], -1))
    yp = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1))
    inter = tf.reduce_sum(yt * yp, axis=1)
    denom = tf.reduce_sum(yt, axis=1) + tf.reduce_sum(yp, axis=1)
    return 1.0 - tf.reduce_mean((2.0 * inter + eps) / (denom + eps))


def combined_loss(y_true, y_pred):
    return weighted_bce(y_true, y_pred) + DICE_WEIGHT * dice_loss(y_true, y_pred)


def dice_coef(y_true, y_pred):
    return 1.0 - dice_loss(y_true, y_pred)


CUSTOM_OBJECTS = {
    "combined_loss": combined_loss,
    "dice_coef":     dice_coef,
}
