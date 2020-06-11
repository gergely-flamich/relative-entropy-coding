import glob
import os

from sacred import Ingredient

import tensorflow as tf


data_ingredient = Ingredient('dataset')


@data_ingredient.config
def data_config():
    dataset_name = "clic2019"

    return_img_name = True

    dataset_base_path = "/scratch/gf332/datasets/"

    training_patch_size = None

    if dataset_name == "clic2019":
        dataset_path = f"{dataset_base_path}/clic2019"

        num_channels = 3

        training_patch_size = 256

        test_split_name = "test"

    elif dataset_name == "kodak":
        dataset_path = f"{dataset_base_path}/kodak"

        num_channels = 3

        training_patch_size = 256

        test_split_name = "test"

    # Can be 'train' or 'test'
    split = "train"


@data_ingredient.capture
def load_dataset(dataset_path,
                 split,
                 training_patch_size,
                 test_split_name,
                 return_image_name=False):
    if split == "test":
        split = test_split_name

    # If we are loading Kodak or CLIC2019
    with tf.device("/CPU:0"):
        dataset = tf.data.Dataset.list_files(f"{dataset_path}/{split}/*.png")
        dataset = dataset.map(read_png, num_parallel_calls=16)

        if split == "train":
            dataset = dataset.map(
                lambda l, x: (l, tf.image.random_crop(x, (training_patch_size, training_patch_size, 3)))
            )

    return dataset


@data_ingredient.capture
def read_png(filename):
    """
    Loads a PNG image file. Taken from Balle's implementation
    """
    image_raw = tf.io.read_file(filename)
    image = tf.image.decode_image(image_raw, channels=3)
    image = tf.cast(image, tf.float32)
    image = image / 255.

    basename = tf.py_function(lambda s: os.path.basename(s.numpy().decode('utf-8')),
                              inp=[filename],
                              Tout=tf.string)

    return basename, image


def quantize_image(image):
  image = tf.round(image * 255)
  image = tf.saturate_cast(image, tf.uint8)
  return image


def write_png(image, filename):
    # Quantize the image first
    image = quantize_image(image)

    image_string = tf.image.encode_png(image)
    tf.io.write_file(filename, image_string)