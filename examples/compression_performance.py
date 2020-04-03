from sacred import Experiment

import os

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
    # Can be "compress" or "initialize" or "update_sampler"
    mode = "compress"

    if mode == "compress" or mode == "update_sampler":
        num_test_images = 1
    elif mode == "initialize":
        num_test_images = 300
        batch_size = 128

    # Model configurations
    model_save_base_dir = "/scratch/gf332/models/relative-entropy-coding"

    model = "resnet_vae"

    train_dataset = "imagenet32"

    kl_per_partition = 10.0

    sampler = "rejection"
    sampler_args = {}

    if sampler == "rejection":
        sampler_args = {
            "sample_buffer_size": 10000,
            "r_buffer_size": 1000000
        }

    elif sampler == "importance":
        sampler_args = {
            "alpha": np.inf,
            "coding_bits": kl_per_partition / np.log(2),

        }

    if model == "vae":
        latent_size = 50

        model_config = {
            "latent_size": latent_size,
            "sampler": sampler
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
            "sampler": sampler,
            "sampler_args": sampler_args,
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
def resnet_vae_initialize(dataset_info,
                          model_config,
                          model_save_dir,
                          num_test_images,
                          batch_size,
                          dataset,
                          _log):
    # -------------------------------------------------------------------------
    # Batch the dataset
    # -------------------------------------------------------------------------
    dataset = dataset.take(num_test_images).batch(batch_size)

    # -------------------------------------------------------------------------
    # Create model
    # -------------------------------------------------------------------------
    model = BidirectionalResNetVAE(**model_config)

    # Initialize_model_weights
    model(tf.zeros((1, 32, 32, dataset_info["num_channels"])))

    # -------------------------------------------------------------------------
    # Create Checkpoints
    # -------------------------------------------------------------------------
    if not os.path.exists(f"{model_save_dir}/compressor_initialized.index"):
        optimizer = tf.optimizers.Adamax()
        ckpt = tf.train.Checkpoint(model=model,
                                   optimizer=optimizer)

        manager = tf.train.CheckpointManager(ckpt, model_save_dir, max_to_keep=10)

        # Restore previous session
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        if manager.latest_checkpoint:
            _log.info(f"Restored model from {manager.latest_checkpoint}")
        else:
            _log.info("Initializing model from scratch.")

        # Swap in Exponential Moving Average shadow variables for evaluation
        model.swap_in_ema_variables()

    else:
        model.load_weights(f"{model_save_dir}/compressor_initialized").expect_partial()

    # -------------------------------------------------------------------------
    # Set-up for compression
    # -------------------------------------------------------------------------
    for images in dataset:
        model.update_coders(images)
    model.save_weights(f"{model_save_dir}/compressor_initialized")


@ex.capture
def resnet_vae_compress(model_config,
                        model_save_dir,
                        num_test_images,
                        update_sampler,
                        dataset,
                        dataset_info,
                        _log):
    # -------------------------------------------------------------------------
    # Batch the dataset
    # -------------------------------------------------------------------------
    dataset = dataset.batch(1).take(num_test_images)

    # -------------------------------------------------------------------------
    # Create model
    # -------------------------------------------------------------------------
    model = BidirectionalResNetVAE(**model_config)

    # Initialize_model_weights
    model(tf.zeros((1, 32, 32, dataset_info["num_channels"])))

    # -------------------------------------------------------------------------
    # Restore model
    # -------------------------------------------------------------------------
    model.load_weights(f"{model_save_dir}/compressor_initialized").expect_partial()

    for images in dataset:
        model(images)

    if dataset_info["dataset_name"] == "clic2019":
        num_pixels = images.shape[1] * images.shape[2]
    else:
        num_pixels = dataset_info["num_pixels"]

    kld = model.kl_divergence(empirical=True, minimum_kl=0.)
    neg_elbo = -model.log_likelihood + kld
    bpp = neg_elbo / (num_pixels * np.log(2))
    bpd = bpp / dataset_info["num_channels"]

    print(f"Negative ELBO: {neg_elbo:.3f}, KL divergence: {kld:.3f}, BPP: {bpp:.5f}, BPD: {bpd:.5f}")

    # -------------------------------------------------------------------------
    # Compress images
    # -------------------------------------------------------------------------
    for images in dataset:
        block_indices, reconstruction = model.compress(images, update_sampler=update_sampler, seed=42)
        if not update_sampler:
            print(f"Negative ELBO: {neg_elbo:.3f}, KL divergence: {kld:.3f}, BPP: {bpp:.5f}, BPD: {bpd:.5f}")
            print("Codelength={}, residuals={}".format(model.get_codelength(block_indices), -model.log_likelihood))
    if update_sampler:
        model.save_weights(f"{model_save_dir}/compressor_initialized")


@ex.automain
def compress_data(model, mode, _log):
    dataset, _ = load_dataset(split="test")

    if model == "vae":
        _log.info("Testing MNIST VAE!")
        test_vae(dataset=dataset)

    elif model == "resnet_vae":
        if mode == "compress":
            _log.info("Compressing using a ResNet VAE!")
            resnet_vae_compress(dataset=dataset, update_sampler=False)
        elif mode == "initialize":
            _log.info("Initializing compressors for a ResNet VAE!")
            resnet_vae_initialize(dataset=dataset)
        elif mode == "update_sampler":
            _log.info("Updating sampler for a ResNet VAE!")
            resnet_vae_compress(dataset=dataset, update_sampler=True)

