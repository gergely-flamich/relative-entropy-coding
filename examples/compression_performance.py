from sacred import Experiment

import os
import json
import datetime
import time

import tensorflow as tf
import numpy as np

from rec.models.resnet_vae import BidirectionalResNetVAE
from rec.models.large_resnet_vae import LargeResNetVAE

from datasets import data_ingredient, load_dataset, write_png
from rec.coding.utils import CodingError

# tf.config.experimental.set_visible_devices([], 'GPU')

ex = Experiment("compression_performance", ingredients=[data_ingredient])


@ex.config
def default_config(dataset_info):
    # Can be "compress" or "initialize" or "update_sampler"
    mode = "compress"

    if mode == "compress" or mode == "update_sampler":
        num_test_images = 1
        output_file = 'results.csv'
        save_reconstructions = True
        reconstruction_dir_name = 'reconstructions'

    elif mode == "initialize":
        num_test_images = 300
        batch_size = 128

    # Model configurations
    model_save_base_dir = "/scratch/gf332/models/relative-entropy-coding"

    model = "resnet_vae"

    train_dataset = "imagenet32"

    kl_per_partition = 3.

    block_size = 1000

    coder_args = {
        "block_size": block_size
    }

    sampler = "rejection"
    sampler_args = {}
    extrapolate_auxiliary_vars = True
    n_beams = 10
    extra_samples = 1.

    if sampler == "rejection":
        sampler_args = {
            "sample_buffer_size": 10000,
            "r_buffer_size": 1000000,
            "extrapolate_auxiliary_vars": extrapolate_auxiliary_vars
        }
    elif sampler == "importance":
        sampler_args = {
            "alpha": np.inf,
            "coding_bits": kl_per_partition / np.log(2),
            "extrapolate_auxiliary_vars": extrapolate_auxiliary_vars
        }
    elif sampler == 'beam_search':
        sampler_args = {
            "n_beams": n_beams,
            "extra_samples": extra_samples,
            "extrapolate_auxiliary_vars": extrapolate_auxiliary_vars
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
            "coder_args": coder_args,
            "kl_per_partition": kl_per_partition
        }

        lamb = 0.1
        beta = 1.

        model_save_dir = f"{model_save_base_dir}/{train_dataset}/{model}/" \
                         f"/{'iaf' if use_iaf else 'gaussian'}/blocks_{num_res_blocks}/" \
                         f"beta_{beta:.3f}_lamb_{lamb:.3f}"

    elif model == "large_resnet_vae":

        num_res_blocks = 2

        model_config = {
            "latent_size": "variable",
            "sampler": sampler,
            "sampler_args": sampler_args,
            "coder_args": coder_args,
            "first_deterministic_filters": 160,
            "first_stochastic_filters": 128,
            "second_deterministic_filters": 160,
            "second_stochastic_filters": 128,
            "kl_per_partition": kl_per_partition
        }

        lamb = 0.1
        beta = 1.

        model_save_dir = f"{model_save_base_dir}/{train_dataset}/{model}/" \
                         f"beta_{beta:.3f}_lamb_{lamb:.3f}"

    compressor_initialized_dir = os.path.join(model_save_dir, "compressor_initialized_{}".format(kl_per_partition))


@ex.capture
def test_vae(dataset):
    pass


@ex.capture
def resnet_vae_initialize(model,
                          dataset_info,
                          model_config,
                          model_save_dir,
                          num_test_images,
                          batch_size,
                          dataset,
                          kl_per_partition,
                          compressor_initialized_dir,
                          _log):
    # -------------------------------------------------------------------------
    # Batch the dataset
    # -------------------------------------------------------------------------
    dataset = dataset.take(num_test_images).batch(batch_size)

    # -------------------------------------------------------------------------
    # Create model
    # -------------------------------------------------------------------------
    if model == "resnet_vae":
        model = BidirectionalResNetVAE(**model_config)

    elif model == "large_resnet_vae":
        model = LargeResNetVAE(**model_config)

    else:
        raise NotImplementedError

    # Initialize_model_weights
    model(tf.zeros((1, 32, 32, dataset_info["num_channels"])))

    # -------------------------------------------------------------------------
    # Create Checkpoints
    # -------------------------------------------------------------------------
    if not os.path.exists(f"{compressor_initialized_dir}/compressor_initialized.index"):
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
        model.load_weights(f"{compressor_initialized_dir}/compressor_initialized").expect_partial()

    # -------------------------------------------------------------------------
    # Set-up for compression
    # -------------------------------------------------------------------------
    model.save_weights(f"{compressor_initialized_dir}/compressor_initialized")
    for images in dataset:
        model.update_coders(images)
    model.save_weights(f"{compressor_initialized_dir}/compressor_initialized")


@ex.capture
def resnet_vae_compress(model,
                        model_config,
                        model_save_dir,
                        num_test_images,
                        update_sampler,
                        dataset,
                        dataset_info,
                        kl_per_partition,
                        extrapolate_auxiliary_vars,
                        output_file,
                        save_reconstructions,
                        reconstruction_dir_name,
                        _log):
    # -------------------------------------------------------------------------
    # Batch the dataset
    # Important note: dataset_info.return_img_name is assumed to be true
    # -------------------------------------------------------------------------
    dataset = dataset.take(num_test_images)

    # -------------------------------------------------------------------------
    # Create model
    # -------------------------------------------------------------------------
    if model == "resnet_vae":
        model = BidirectionalResNetVAE(**model_config)

    elif model == "large_resnet_vae":
        model = LargeResNetVAE(**model_config)

    else:
        raise NotImplementedError

    # Initialize_model_weights
    model(tf.zeros((1, 32, 32, dataset_info["num_channels"])))

    # -------------------------------------------------------------------------
    # Restore model
    # -------------------------------------------------------------------------
    if extrapolate_auxiliary_vars:
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
        model_dir = model_save_dir
    else:
        model_dir = os.path.join(model_save_dir, "compressor_initialized_{}".format(kl_per_partition))
        model.load_weights(f"{model_dir}/compressor_initialized").expect_partial()

    # -------------------------------------------------------------------------
    # Compress images
    # -------------------------------------------------------------------------
    output_filename = os.path.join(model_dir, output_file)
    with open(output_filename, "w") as outfile:
        outfile.write(', '.join(['name',
                                 'residual',
                                 'KL',
                                 'lossless_BPP',
                                 'lossy_BPP',
                                 'BPD',
                                 'comp_residual',
                                 'comp_codelength',
                                 'comp_KL',
                                 'comp_lossless_BPP',
                                 'comp_lossy_BPP',
                                 'comp_BPD',
                                 'comp_time',
                                 'ideal_PSNR',
                                 'ideal_MS_SSIM',
                                 'PSNR',
                                 'MS_SSIM']))
        outfile.write('\n')

    if update_sampler:
        for _, image in dataset:
            model.compress(image[None, :], update_sampler=update_sampler, seed=42)
        model.save_weights(f"{model_dir}/compressor_initialized_sampler_updated")
        return

    for image_name, image in dataset:

        # Image name is originally stored as a TF bytestring
        image_name = image_name.numpy().decode('utf-8')
        print(f"Compressing {image_name}!")

        # Measurements without compression
        ideal_reconstruction = model(image[None, :])

        ideal_psnr = tf.image.psnr(ideal_reconstruction,
                                   image + 0.5,
                                   max_val=1.0)[0]

        ideal_ms_ssim = tf.image.ssim_multiscale(ideal_reconstruction,
                                                 image + 0.5,
                                                 max_val=1.0)[0]

        if dataset_info["dataset_name"] == "clic2019":
            num_pixels = image.shape[1] * image.shape[2]
        else:
            num_pixels = dataset_info["num_pixels"]
        kld = model.kl_divergence(empirical=False, minimum_kl=0.)
        residual = -model.log_likelihood
        lossless_bpp = (kld + residual) / (num_pixels * np.log(2))
        bpd = lossless_bpp / dataset_info["num_channels"]

        lossy_bpp = kld / (num_pixels * np.log(2))

        # Measurements with compression
        start_time = time.time()
        try:
            block_indices, reconstruction = model.compress(image[None, :], update_sampler=update_sampler, seed=42)
        except CodingError:
            _log.info("Coding Error occurred.")
            with open(output_filename, "a") as outfile:
                outfile.write(', '.join([image_name] +
                                        [str(float(v)) for v in [residual,
                                                                 kld,
                                                                 lossless_bpp,
                                                                 lossy_bpp,
                                                                 bpd,
                                                                 -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]))
                outfile.write('\n')
            continue
        comp_time = time.time() - start_time
        comp_kld = model.kl_divergence(empirical=False, minimum_kl=0.)
        comp_codelength = model.get_codelength(block_indices)
        comp_residual = -model.log_likelihood
        comp_lossless_bpp = (comp_kld + comp_residual) / (num_pixels * np.log(2))
        comp_lossy_bpp = comp_kld / (num_pixels * np.log(2))
        comp_bpd = comp_lossless_bpp / dataset_info["num_channels"]

        psnr = tf.image.psnr(reconstruction, image + 0.5, max_val=1.0)[0]
        ms_ssim = tf.image.ssim_multiscale(reconstruction, image + 0.5, max_val=1.0)[0]

        _log.info(f"KL divergence: {kld:.3f}, "
                  f"residuals: {residual:.3f}, "
                  f"lossless BPP: {lossless_bpp:.5f}, "
                  f"BPD: {bpd:.5f}, "
                  f"ideal lossy BPP: {lossy_bpp}, "
                  f"actual lossy BPP: {comp_lossy_bpp}, "
                  f"ideal PSNR: {ideal_psnr:.4f}, "
                  f"ideal MS-SSIM: {ideal_ms_ssim:.4f}, "
                  f"PSNR: {psnr:.4f}, "
                  f"MS-SSIM: {ms_ssim:.4f},"
                  f"comp_time: {comp_time}")
        _log.info("Codelength: {}, residuals: {}".format(comp_codelength, comp_residual))
        with open(output_filename, "a") as outfile:
            outfile.write(', '.join([image_name] +
                                    [str(float(v)) for v in [residual,
                                                             kld,
                                                             lossless_bpp,
                                                             lossy_bpp,
                                                             bpd,
                                                             comp_residual,
                                                             comp_codelength,
                                                             comp_kld,
                                                             comp_lossless_bpp,
                                                             comp_lossy_bpp,
                                                             comp_bpd,
                                                             comp_time,
                                                             ideal_psnr,
                                                             ideal_ms_ssim,
                                                             psnr,
                                                             ms_ssim]]))
            outfile.write('\n')

        if save_reconstructions:
            print("saving")
            write_png(reconstruction[0], f"{model_dir}/"
                                         f"{reconstruction_dir_name}/"
                                         f"{image_name}")

@ex.automain
def compress_data(model, mode, _log):
    dataset, _ = load_dataset(split="test",
                              return_image_name=mode == "compress")

    if model == "vae":
        _log.info("Testing MNIST VAE!")
        test_vae(dataset=dataset)

    elif model in ["resnet_vae", "large_resnet_vae"]:
        if mode == "compress":
            _log.info("Compressing using a ResNet VAE!")
            resnet_vae_compress(dataset=dataset, update_sampler=False)
        elif mode == "initialize":
            _log.info("Initializing compressors for a ResNet VAE!")
            resnet_vae_initialize(dataset=dataset)
        elif mode == "update_sampler":
            _log.info("Updating sampler for a ResNet VAE!")
            resnet_vae_compress(dataset=dataset, update_sampler=True)
