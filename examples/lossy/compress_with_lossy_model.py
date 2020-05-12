from sacred import Experiment

import os
import time

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from rec.models.lossy import Large2LevelVAE, Large1LevelVAE
from rec.coding import GaussianCoder, BeamSearchCoder
from rec.coding.samplers import RejectionSampler, ImportanceSampler

from datasets import data_ingredient, load_dataset, write_png

tfd = tfp.distributions
#tf.config.experimental.set_visible_devices([], 'GPU')

ex = Experiment("compress_with_loss_model", ingredients=[data_ingredient])


@ex.config
def config(dataset):
    model_dir = ""

    model = "large_level_1_vae"

    if model == "large_level_1_vae":
        num_filters = 192

        model_config = {
            "num_filters": num_filters,
        }

    elif model == "large_level_2_vae":
        level_1_filters = 196
        level_2_filters = 128

        model_config = {
            "level_1_filters": level_1_filters,
            "level_2_filters": level_2_filters
        }

    compression_seed = 42
    num_test_images = 1

    output_base_dir = f'{dataset["dataset_name"]}/'
    output_file = f'{output_base_dir}/results.csv'
    save_reconstructions = True
    reconstruction_dir_name = f'{output_base_dir}/reconstructions'

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
        }
    elif sampler == "importance":
        sampler_args = {
            "alpha": np.inf,
            "coding_bits": kl_per_partition / np.log(2),
        }
    elif sampler == 'beam_search':
        sampler_args = {
            "n_beams": n_beams,
            "extra_samples": extra_samples,
        }


@ex.automain
def compress(dataset,

             model,
             model_config,
             model_dir,

             output_file,
             save_reconstructions,
             compression_seed,
             reconstruction_dir_name,
             kl_per_partition,

             sampler,
             sampler_args,
             coder_args,
             block_size,

             num_test_images,

             _log):
    if len(model_dir) == 0:
        raise ValueError("Model directory must be specified!")

    # -------------------------------------------------------------------------
    # Create sampler
    # -------------------------------------------------------------------------
    if sampler == "rejection":
        coder = GaussianCoder(sampler=RejectionSampler(**sampler_args),
                              kl_per_partition=kl_per_partition,
                              **coder_args)
    elif sampler == "importance":
        # Setting alpha=inf will select the sample with
        # the best importance weights
        coder = GaussianCoder(sampler=ImportanceSampler(**sampler_args),
                              kl_per_partition=kl_per_partition,
                              **coder_args)
    elif sampler == "beam_search":
        coder = BeamSearchCoder(kl_per_partition=kl_per_partition,
                                n_beams=sampler_args['n_beams'],
                                extra_samples=sampler_args['extra_samples'],
                                **coder_args)
    else:
        raise ValueError("Sampler must be one of ['rejection', 'importance', 'beam_search'],"
                         f"but got {sampler}!")

    # -------------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------------
    ds = load_dataset(return_image_name=True).take(num_test_images)

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------
    model = {
        "large_level_1_vae": Large1LevelVAE,
        "large_level_2_vae": Large2LevelVAE,
    }[model](**model_config)

    # Initialize model
    model.build(input_shape=(1, 256, 256, 3))

    # -------------------------------------------------------------------------
    # Restore model
    # -------------------------------------------------------------------------
    ckpt = tf.train.Checkpoint(model=model)

    manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=10)

    # Restore previous session
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    if manager.latest_checkpoint:
        _log.info(f"Restored model from {manager.latest_checkpoint}")
    else:
        _log.info("Initializing model from scratch.")

    # -------------------------------------------------------------------------
    # Compress images
    # -------------------------------------------------------------------------
    os.makedirs(f"{model_dir}/{reconstruction_dir_name}", exist_ok=True)
    output_filename = os.path.join(model_dir, output_file)
    with open(output_filename, "w") as outfile:
        outfile.write(', '.join(['name',
                                 'KL',
                                 'lossy_BPP',
                                 'comp_codelength',
                                 'comp_lossy_BPP',
                                 'comp_time',
                                 'ideal_PSNR',
                                 'ideal_MS_SSIM',
                                 'PSNR',
                                 'MS_SSIM']))
        outfile.write('\n')

    for image_name, image in ds:

        original_image_shape = image.shape
        h, w, c = original_image_shape

        # We slightly upscale the image so that its width and height are powers of 2
        image = tf.image.resize(image,
                                [h - (h % 64), w - (w % 64)])

        # Image name is originally stored as a TF bytestring
        image_name = image_name.numpy().decode('utf-8')
        if image_name.startswith('img_Tensor'):
            image_name = dataset['dataset_name'] + '_sample'
        print(f"Compressing {image_name}!")

        # Measurements without compression
        ideal_reconstruction = model(image[None, :])

        ideal_reconstruction = tf.clip_by_value(ideal_reconstruction, 0., 1.)

        ideal_psnr = tf.squeeze(tf.image.psnr(ideal_reconstruction * 255.,
                                              image * 255.,
                                              max_val=255.))

        ideal_ms_ssim = tf.squeeze(tf.image.ssim_multiscale(ideal_reconstruction * 255.,
                                                            image * 255.,
                                                            max_val=255.))

        num_pixels = w * h

        klds = model.kl_divergence()
        kld = tf.reduce_sum(klds)

        lossy_bpp = kld / (num_pixels * np.log(2))

        _log.info(f"KL divergence: {kld:.3f}, "
                  f"ideal lossy BPP: {lossy_bpp}, "
                  f"ideal PSNR: {ideal_psnr:.4f}, "
                  f"ideal MS-SSIM: {ideal_ms_ssim:.4f}, ")

        # Measurements with compression
        start_time = time.time()
        try:
            compressed_file_path = f"{model_dir}/{reconstruction_dir_name}/{image_name}.rec"

            reconstruction = model.compress(file_path=compressed_file_path,
                                            image=image,
                                            seed=compression_seed,
                                            sampler=coder,
                                            block_size=block_size,
                                            max_index=20)

            compressed_file_bits = os.path.getsize(compressed_file_path) * 8

            # reconstruction_ = model.decompress(file_path=compressed_file_path,
            #                                    sampler=coder)

        except Exception:
            _log.info("Coding Error occurred.")
            raise

        comp_time = time.time() - start_time
        comp_lossy_bpp = compressed_file_bits / num_pixels
        reconstruction = tf.clip_by_value(reconstruction, 0., 1.)

        psnr = tf.squeeze(tf.image.psnr(reconstruction * 255.,
                                        image * 255.,
                                        max_val=255.))

        ms_ssim = tf.squeeze(tf.image.ssim_multiscale(reconstruction * 255.,
                                                      image * 255.,
                                                      max_val=255.))

        _log.info(f"KL divergence: {kld:.3f}, "
                  f"ideal lossy BPP: {lossy_bpp}, "
                  f"actual lossy BPP: {comp_lossy_bpp}, "
                  f"ideal PSNR: {ideal_psnr:.4f}, "
                  f"ideal MS-SSIM: {ideal_ms_ssim:.4f}, "
                  f"PSNR: {psnr:.4f}, "
                  f"MS-SSIM: {ms_ssim:.4f},"
                  f"comp_time: {comp_time}")
        _log.info(f"Codelength: {compressed_file_bits}")

        with open(output_filename, "a") as outfile:
            outfile.write(', '.join([image_name] +
                                    [str(float(v)) for v in [kld,
                                                             lossy_bpp,
                                                             compressed_file_bits,
                                                             comp_lossy_bpp,
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
