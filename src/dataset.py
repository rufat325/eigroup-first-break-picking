"""tf.data input pipelines for training and validation."""
import numpy as np
import tensorflow as tf

from .config import PATCH_WIDTH, PATCH_HEIGHT, BATCH_SIZE, SPLIT_SEED


def _load_patch_numpy(path):
    data = np.load(path.decode("utf-8"))
    img  = data["image"].astype(np.float32)[..., None]
    mask = data["mask"].astype(np.float32)[..., None]
    return img, mask


def _tf_load_patch(path):
    img, mask = tf.numpy_function(
        _load_patch_numpy, [path], (tf.float32, tf.float32),
    )
    img.set_shape((PATCH_WIDTH, PATCH_HEIGHT, 1))
    mask.set_shape((PATCH_WIDTH, PATCH_HEIGHT, 1))
    return img, mask


def _augment(img, mask):
    # Horizontal flip across the trace axis.
    if tf.random.uniform(()) < 0.5:
        img  = tf.reverse(img,  axis=[0])
        mask = tf.reverse(mask, axis=[0])
    # Random amplitude scaling (sign-preserving).
    img = img * tf.random.uniform((), 0.7, 1.3)
    # Polarity flip; first-break position is sign-invariant.
    if tf.random.uniform(()) < 0.5:
        img = -img
    return img, mask


def build_dataset(paths, shuffle=False, augment=False, batch_size=BATCH_SIZE):
    """Construct a tf.data.Dataset from a list of .npz patch paths."""
    ds = tf.data.Dataset.from_tensor_slices(paths)
    if shuffle:
        ds = ds.shuffle(len(paths), seed=SPLIT_SEED, reshuffle_each_iteration=True)
    ds = ds.map(_tf_load_patch, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
