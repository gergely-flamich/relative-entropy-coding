import glob

from sacred import Ingredient

import tensorflow_datasets as tfds
import tensorflow as tf


data_ingredient = Ingredient('dataset_info')


@data_ingredient.config
def data_config():
    dataset_name = "mnist"

    dataset_base_path = "/scratch/gf332/datasets/"

    training_patch_size = None

    if dataset_name == "mnist":
        tfds_name = "mnist"
        dataset_path = f"{dataset_base_path}/mnist"

        normalizer = 1.
        num_pixels = 28 * 28
        num_channels = 1

        test_split_name = "test"

    elif dataset_name == "binarized_mnist":
        tfds_name = "binarized_mnist"
        dataset_path = f"{dataset_base_path}/binarized_mnist"

        normalizer = 1.
        num_pixels = 28 * 28
        num_channels = 1

        test_split_name = "test"

    elif dataset_name == "cifar10":
        tfds_name = "cifar10"
        dataset_path = f"{dataset_base_path}/cifar10"

        normalizer = 256.
        num_pixels = 32 * 32
        num_channels = 3

        test_split_name = "test"

    elif dataset_name == "imagenet32":
        tfds_name = "downsampled_imagenet/32x32"
        dataset_path = f"{dataset_base_path}/imagenet32"

        normalizer = 256.
        num_pixels = 32 * 32
        num_channels = 3

        test_split_name = "validation"

    elif dataset_name == "imagenet64":
        tfds_name = "downsampled_imagenet/64x64"
        dataset_path = f"{dataset_base_path}/imagenet64"

        normalizer = 256.
        num_pixels = 64 * 64
        num_channels = 3

        test_split_name = "validation"

    elif dataset_name == "clic2019":
        tfds_name = None
        dataset_path = f"{dataset_base_path}/clic2019"

        normalizer = 256.
        num_pixels = -1
        num_channels = 3

        training_patch_size = 256

        test_split_name = "test"

    elif dataset_name == "kodak":
        tfds_name = None
        dataset_path = f"{dataset_base_path}/kodak"

        normalizer = 256.
        num_pixels = 768 * 512
        num_channels = 3

        training_patch_size = 256

        test_split_name = "test"

    # Can be 'train' or 'test'
    split = "train"


@data_ingredient.capture
def load_dataset(tfds_name,
                 dataset_path,
                 split,
                 normalizer,
                 training_patch_size,
                 test_split_name):

    # If we are loading Kodak or CLIC2019
    if tfds_name is None:
        with tf.device("/CPU:0"):

            if split == "test":
                split = test_split_name

            files = glob.glob(f"{dataset_path}/{split}/*.png")

            if not files:
                raise RuntimeError("No training images found at '{}'.".format(dataset_path))

            dataset = tf.data.Dataset.from_tensor_slices(files)
            dataset = dataset.map(
                read_png, num_parallel_calls=16)

            if split == "train":
                dataset = dataset.map(
                    lambda x: tf.image.random_crop(x, (training_patch_size, training_patch_size, 3)))

                num_pixels = training_patch_size * training_patch_size

        ds = dataset

    else:
        ds = tfds.load(tfds_name,
                       data_dir=dataset_path)
        ds = ds[split]
        ds = ds.map(lambda image: tf.cast(image["image"], tf.float32))

    def prepare(image):
        image = (image + 0.5) / normalizer
        image = tf.clip_by_value(image, 0., 1.) - 0.5

        return image

    ds = ds.map(prepare)

    return ds


def read_png(filename):
    """
    Loads a PNG image file. Taken from Balle's implementation
    """
    image_raw = tf.io.read_file(filename)
    image = tf.image.decode_image(image_raw, channels=3)
    image = tf.cast(image, tf.float32)

    return image
