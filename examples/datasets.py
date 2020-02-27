from sacred import Ingredient

import tensorflow_datasets as tfds
import tensorflow as tf


data_ingredient = Ingredient('dataset_info')


@data_ingredient.config
def data_config():
    tfds_name = "mnist"
    dataset_path = "/scratch/gf332/datasets/mnist"

    # Can be 'train' or 'test'
    split = "train"

    num_pixels = 28 * 28
    num_channels = 1

    normalize = True


@data_ingredient.named_config
def binarized_mnist():
    tfds_name = "binarized_mnist"
    dataset_path = "/scratch/gf332/datasets/mnist"

    normalize = False


@data_ingredient.named_config
def cifar_10():
    tfds_name = "cifar10"
    dataset_path = "/scratch/gf332/datasets/cifar10"

    num_pixels = 32 * 32
    num_channels = 3

    normalize = True


@data_ingredient.capture
def load_dataset(tfds_name, dataset_path, split, normalize, num_pixels, num_channels):

    ds = tfds.load(tfds_name,
                   data_dir=dataset_path)

    ds = ds[split]

    normalizer = 255. if normalize else 1.

    ds = ds.map(lambda x: tf.cast(x["image"], tf.float32) / normalizer)

    return ds, num_pixels, num_channels
