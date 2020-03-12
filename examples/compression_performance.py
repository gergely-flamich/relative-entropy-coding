from sacred import Experiment

import json
import datetime

import tensorflow as tf
import numpy as np

from rec.models.mnist_vae import MNISTVAE
from rec.models.resnet_vae import BidirectionalResNetVAE

from datasets import data_ingredient, load_dataset

tf.config.experimental.set_visible_devices([], 'GPU')

ex = Experiment("compression_performance", ingredients=[data_ingredient])


@ex.config
def default_config(dataset_info):

    num_test_images = 1

    # Model configurations
    model_save_base_dir = "/scratch/gf332/models/relative-entropy-coding"

    model = "resnet_vae"

    train_dataset = "mnist"

    kl_per_partition = 10.0

    if model == "vae":
        latent_size = 50

        model_config = {
            "latent_size": latent_size
        }

        lamb = 0.
        beta = 1.

        model_save_dir = f"{model_save_base_dir}/{train_dataset}/{model}/" \
                         f"latents_{latent_size}_beta_{beta:.3f}_lamb_{lamb:.3f}"

    elif model == "resnet_vae":

        use_iaf = False
        num_res_blocks = 4

        model_config = {
            "use_iaf": use_iaf,
            "latent_size": "variable",
            "num_res_blocks": num_res_blocks,
            "deterministic_filters": 160,
            "stochastic_filters": 32,
        }

        lamb = 0.1
        beta = 1.

        model_save_dir = f"{model_save_base_dir}/{train_dataset}/{model}/" \
                         f"/{'iaf' if use_iaf else 'gaussian'}/blocks_{num_res_blocks}/" \
                         f"beta_{beta:.3f}_lamb_{lamb:.3f}"


@ex.capture
def test_vae(dataset):
    pass


@ex.capture
def test_resnet_vae(model_config,
                    model_save_dir,
                    num_test_images,
                    dataset,
                    kl_per_partition,
                    test_dataset_name,
                    num_pixels,
                    num_channels,
                    _log):

    # -------------------------------------------------------------------------
    # Batch the dataset
    # -------------------------------------------------------------------------
    dataset = dataset.batch(num_test_images).take(1)

    # -------------------------------------------------------------------------
    # Create model
    # -------------------------------------------------------------------------
    model = BidirectionalResNetVAE(**model_config)

    # Initialize_model_weights
    model(tf.zeros((1, 32, 32, num_channels)))

    # -------------------------------------------------------------------------
    # Create Checkpoints
    # -------------------------------------------------------------------------
    optimizer = tf.optimizers.Adamax()
    ckpt = tf.train.Checkpoint(model=model,
                               optimizer=optimizer)

    manager = tf.train.CheckpointManager(ckpt, model_save_dir, max_to_keep=3)

    # Restore previous session
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        _log.info(f"Restored model from {manager.latest_checkpoint}")
    else:
        _log.info("Initializing model from scratch.")

    # Swap in Exponential Moving Average shadow variables for evaluation
    model.swap_in_ema_variables()

    for images in dataset:
        model(images)

    if test_dataset_name == "clic2019":
        num_pixels = images.shape[1] * images.shape[2]

    kld = model.kl_divergence(empirical=True, minimum_kl=0.)
    neg_elbo = -model.log_likelihood + kld
    bpp = neg_elbo / (num_pixels * np.log(2))
    bpd = bpp / num_channels

    print(f"Negative ELBO: {neg_elbo:.3f}, KL divergence: {kld:.3f}, BPP: {bpp:.5f}, BPD: {bpd:.5f}")

    # -------------------------------------------------------------------------
    # Set-up for compression
    # -------------------------------------------------------------------------
    for images in dataset:
        model.initialize_coders(images, kl_per_partition=kl_per_partition)

    # -------------------------------------------------------------------------
    # Compress images
    # -------------------------------------------------------------------------
    for images in dataset:
        block_indices, reconstruction = model.compress(images, seed=42)


@ex.automain
def compress_data(model, _log):
    dataset, num_pixels, num_channels, test_dataset_name = load_dataset(split="test")

    if model == "vae":
        _log.info("Testing MNIST VAE!")
        test_vae(dataset=dataset)

    elif model == "resnet_vae":
        _log.info("Testing a ResNet VAE!")
        test_resnet_vae(dataset=dataset,
                        test_dataset_name=test_dataset_name,
                        num_pixels=num_pixels,
                        num_channels=num_channels)