from sacred import Ingredient

import tensorflow_datasets as tfds
import tensorflow as tf


data_ingredient = Ingredient('dataset_info')


@data_ingredient.config
def data_config():
    dataset_name = "mnist"

    dataset_base_path = "/scratch/gf332/datasets/"

    if dataset_name == "mnist":
        tfds_name = "mnist"
        dataset_path = f"{dataset_base_path}/mnist"

        normalize = False
        num_pixels = 28 * 28
        num_channels = 1

    elif dataset_name == "binarized_mnist":
        tfds_name = "binarized_mnist"
        dataset_path = f"{dataset_base_path}/binarized_mnist"

        normalize = False
        num_pixels = 28 * 28
        num_channels = 1

    elif dataset_name == "cifar10":
        tfds_name = "cifar10"
        dataset_path = f"{dataset_base_path}/cifar10"

        normalize = True
        num_pixels = 32 * 32
        num_channels = 3

    elif dataset_name == "imagenet32":
        tfds_name = "downsampled_imagenet/32x32"
        dataset_path = f"{dataset_base_path}/imagenet32"

        normalize = True
        num_pixels = 32 * 32
        num_channels = 3

    elif dataset_name == "imagenet64":
        tfds_name = "downsampled_imagenet/64x64"
        dataset_path = f"{dataset_base_path}/imagenet64"

        normalize = True
        num_pixels = 64 * 64
        num_channels = 3

    # Can be 'train' or 'test'
    split = "train"


@data_ingredient.capture
def load_dataset(tfds_name, dataset_path, split, normalize, num_pixels, num_channels):

    ds = tfds.load(tfds_name,
                   data_dir=dataset_path)

    ds = ds[split]

    normalizer = 256. if normalize else 1.

    def prepare(image):
        image = tf.cast(image["image"], tf.float32)

        image = (image + 0.5) / normalizer

        image = tf.clip_by_value(image, 0., 1.) - 0.5

        return image

    ds = ds.map(prepare)

    return ds, num_pixels, num_channels
