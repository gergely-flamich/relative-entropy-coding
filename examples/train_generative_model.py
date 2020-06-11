from sacred import Experiment
from sacred.stflow import LogFileWriter

import json
import datetime

import tensorflow as tf
import tensorflow_probability as tfp

from rec.models.mnist_vae import MNISTVAE
from rec.models.resnet_vae import BidirectionalResNetVAE
from rec.models.large_resnet_vae_new import LargeResNetVAE

from datasets import data_ingredient, load_dataset

tfs = tf.summary
tfd = tfp.distributions

log_2 = tf.math.log(2.)

ex = Experiment('train_generative_model', ingredients=[data_ingredient])


def print_dict(d):
    items = []

    for k, v in d.items():
        items.append(f"{k}-" + (f"{v:.4f}" if type(v) == float else f"{v}"))

    return '.'.join(items)


@ex.config
def default_config(dataset_info):
    # Model configurations
    model_save_base_dir = "/scratch/gf332/models/relative-entropy-coding"

    model = "resnet_vae"
    optimizer = "adamax"

    lossy = False

    if lossy:
        # Average Bits per pixel
        target_bpp = 0.1

        bpp_buffer_size = 30

        # Start adjusting Beta after the given number of iterations
        adjust_beta_after_iters = 100000

        # Beta will be adjusted until iteration adjust_beta_after_iters + stop_adjust_after_iters
        stop_adjust_after_iters = 50000

    if model == "vae":
        latent_size = 50

        model_config = {
            "latent_size": latent_size
        }

        learning_rate = 3e-4
        lamb = 0.
        beta = 1.

        model_save_dir = f"{model_save_base_dir}/{dataset_info['dataset_name']}/{model}/" \
                         f"latents_{latent_size}_beta_{beta:.3f}_lamb_{lamb:.3f}"

    elif model == "resnet_vae":

        use_iaf = False
        num_res_blocks = 4
        likelihood_function = "discretized_logistic"
        learn_likelihood_scale = True

        sampler_args = {
            "alpha": float('inf'),
            "coding_bits": 10,
        }

        model_config = {
            "sampler": "importance",
            "sampler_args": sampler_args,
            "use_iaf": use_iaf,
            "latent_size": "variable",
            "num_res_blocks": num_res_blocks,
            "deterministic_filters": 160,
            "stochastic_filters": 32,
            "likelihood_function": likelihood_function,
            "learn_likelihood_scale": learn_likelihood_scale
        }

        learning_rate = 1e-3
        lamb = 0.1
        beta = 1.

        model_save_dir = f"{model_save_base_dir}/{dataset_info['dataset_name']}/{model}/" \
                         f"/{'iaf' if use_iaf else 'gaussian'}/blocks_{num_res_blocks}/" \
                         f"beta_{beta:.3f}_lamb_{lamb:.3f}_{likelihood_function}"

        model_save_dir += f"_target_bpp_{target_bpp:.3f}" if lossy else "_lossless"

    elif model == "large_resnet_vae":

        optimizer = "adam"
        distribution = "gaussian"
        likelihood_function = "laplace"
        learn_likelihood_scale = False
        use_gdn = True
        use_sig_convs = True

        lossy = True

        sampler_args = {
            "alpha": float('inf'),
            "coding_bits": 10,
            "n_beams": 10,
            "extra_samples": 1.,
            "extrapolate_auxiliary_vars": True,
        }

        model_config = {
            "use_gdn": use_gdn,
            "use_sig_convs": use_sig_convs,
            "distribution": distribution,
            "sampler": "beam_search",
            "sampler_args": sampler_args,
            "latent_size": "variable",
            "first_deterministic_filters": 196,
            "first_stochastic_filters": 196,
            "second_deterministic_filters": 128,
            "second_stochastic_filters": 128,
            "likelihood_function": likelihood_function,
            "learn_likelihood_scale": learn_likelihood_scale,
        }

        learning_rate = 1e-3
        lamb = 0.01
        beta = 1.

        model_save_dir = f"{model_save_base_dir}/{dataset_info['dataset_name']}/{model}/{distribution}/" \
                         f"beta_{beta:.3f}_lamb_{lamb:.3f}_{likelihood_function}"

        model_save_dir += f"_target_bpp_{target_bpp:.3f}" if lossy else "_lossless"

    # Training-time configurations
    iters = 3000000

    shuffle_buffer_size = 5000
    batch_size = 8
    num_prefetch = 32

    # ELBO related stuff
    beta = 1.
    anneal = False  # Whether to anneal Beta at the start
    annealing_end = 100000  # Steps after which beta is fixed
    drop_learning_rate_after_iter = 50000
    learning_rate_drop_rate = 0.3

    # Logging
    tensorboard_log_freq = 1000

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"{model_save_dir}/logs/{current_time}/train"


@ex.capture
@LogFileWriter(ex)
def train_vae(dataset,
              model_save_dir,
              log_dir,
              tensorboard_log_freq,
              model_config,
              batch_size,
              shuffle_buffer_size,
              num_prefetch,
              learning_rate,
              iters,
              beta,
              annealing_end,
              drop_learning_rate_after_iter,
              learning_rate_drop_rate,
              _log):
    # -------------------------------------------------------------------------
    # Prepare the dataset
    # -------------------------------------------------------------------------
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(num_prefetch)

    # -------------------------------------------------------------------------
    # Create model
    # -------------------------------------------------------------------------
    latent_size = model_config["latent_size"]
    model = MNISTVAE(prior=tfd.Normal(loc=tf.zeros(latent_size),
                                      scale=tf.ones(latent_size)))

    # -------------------------------------------------------------------------
    # Create Optimizer
    # -------------------------------------------------------------------------
    learn_rate = tf.Variable(learning_rate)
    optimizer = tf.optimizers.Adam(learn_rate)

    # -------------------------------------------------------------------------
    # Create Checkpoints
    # -------------------------------------------------------------------------
    ckpt = tf.train.Checkpoint(step=tf.Variable(1, dtype=tf.int64),
                               learn_rate=learn_rate,
                               model=model,
                               optimizer=optimizer)

    manager = tf.train.CheckpointManager(ckpt, model_save_dir, max_to_keep=3)

    # -------------------------------------------------------------------------
    # Create Summary Writer
    # -------------------------------------------------------------------------
    summary_writer = tfs.create_file_writer(log_dir)

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------

    def train_step():
        pass

    # Initialize the model weights
    model(tf.zeros([1, 28, 28, 1]))

    # Restore previous session
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        _log.info(f"Restored model from {manager.latest_checkpoint}")
    else:
        _log.info("Initializing model from scratch.")

    for batch in dataset.take(iters - int(ckpt.step)):

        # Increment the training step
        ckpt.step.assign_add(1)

        # Decrease learning rate after a while
        # if int(ckpt.step) == drop_learning_rate_after_iter:
        #    learn_rate.assign(learning_rate_after_drop)

        with tf.GradientTape() as tape:

            reconstruction = model(batch, training=True)

            log_prob = model.likelihood.log_prob(batch)

            # Get the empirical log-likelihood per image
            nll = -tf.reduce_mean(tf.reduce_sum(log_prob, axis=[1, 2]))

            # Get the empirical KL per latent code
            kl_divergence = tf.reduce_mean(tf.reduce_sum(model.kl_divergence, axis=1))

            # Linearly annealed beta
            beta = tf.minimum(beta, tf.cast(ckpt.step / annealing_end, tf.float32))

            loss = nll + beta * kl_divergence

        if tf.reduce_min(-log_prob) > 10:
            _log.info("Mispredicted pixel!")

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if int(ckpt.step) % tensorboard_log_freq == 0:
            # Get empirical posterior and prior log likelihoods for summaries
            post_log_lik = tf.reduce_mean(tf.reduce_sum(model.post_log_liks, axis=1))
            prior_log_lik = tf.reduce_mean(tf.reduce_sum(model.prior_log_liks, axis=1))

            expected_max_kl = tf.reduce_mean(tf.reduce_max(model.kl_divergence, axis=1))

            # Save model
            save_path = manager.save()
            _log.info(f"Step {int(ckpt.step)}: Saved model to {save_path}")

            with summary_writer.as_default():
                tfs.scalar(name="Loss", data=loss, step=ckpt.step)
                tfs.scalar(name="NLL", data=nll, step=ckpt.step)
                tfs.scalar(name="Posterior_LL", data=post_log_lik, step=ckpt.step)
                tfs.scalar(name="Prior_LL", data=prior_log_lik, step=ckpt.step)
                tfs.scalar(name="KL", data=kl_divergence, step=ckpt.step)

                tfs.scalar(name="Beta", data=beta, step=ckpt.step)

                tfs.image(name="Original", data=batch, step=ckpt.step)
                tfs.image(name="Reconstruction", data=reconstruction, step=ckpt.step)

                tfs.scalar(name="Expected_Max_KL", data=expected_max_kl, step=ckpt.step)


@ex.capture
def train_resnet_vae(dataset,
                     model,

                     dataset_info,
                     model_save_dir,
                     log_dir,
                     tensorboard_log_freq,
                     model_config,
                     batch_size,
                     shuffle_buffer_size,
                     num_prefetch,
                     optimizer,
                     learning_rate,
                     iters,
                     beta,
                     lamb,
                     anneal,
                     annealing_end,
                     drop_learning_rate_after_iter,
                     learning_rate_drop_rate,
                     num_pixels,
                     _log,
                     lossy,
                     target_bpp=None,
                     bpp_buffer_size=None,
                     adjust_beta_after_iters=None,
                     stop_adjust_after_iters=None):
    # -------------------------------------------------------------------------
    # Prepare the dataset
    # -------------------------------------------------------------------------
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(num_prefetch)

    num_channels = dataset_info["num_channels"]

    # -------------------------------------------------------------------------
    # Create model
    # -------------------------------------------------------------------------
    if model == "resnet_vae":
        model = BidirectionalResNetVAE(**model_config)

    elif model == "large_resnet_vae":
        model = LargeResNetVAE(**model_config)

    else:
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Create Optimizer
    # -------------------------------------------------------------------------
    learn_rate = tf.Variable(learning_rate)

    # Initialize optimizer with appropriate learning rate
    optimizer = {
        "adamax": tf.optimizers.Adamax,
        "adam": tf.optimizers.Adam,
    }[optimizer](learn_rate)

    # Initialize the model weights
    for first_pass in dataset.take(1):
        model(first_pass)

    beta = tf.Variable(beta, dtype=tf.float32, name="beta")
    # -------------------------------------------------------------------------
    # Create Checkpoints
    # -------------------------------------------------------------------------
    ckpt = tf.train.Checkpoint(step=tf.Variable(1, dtype=tf.int64),
                               model=model,
                               learn_rate=learn_rate,
                               optimizer=optimizer,
                               beta=beta)

    manager = tf.train.CheckpointManager(ckpt, model_save_dir, max_to_keep=3)

    # -------------------------------------------------------------------------
    # Create Summary Writer
    # -------------------------------------------------------------------------
    summary_writer = tfs.create_file_writer(log_dir)

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------

    @tf.function
    def train_step(batch, model, beta):
        print("Retracing train step!")

        with tf.GradientTape() as tape:

            reconstruction = model(batch, training=True)

            log_likelihood = model.log_likelihood
            kld = model.kl_divergence(empirical=True, minimum_kl=lamb)

            bpp = kld / (num_pixels * log_2)

            if lossy and int(ckpt.step) > adjust_beta_after_iters:
                if bpp > target_bpp + 1e-2:
                    beta.assign(beta * 1.001)

                elif bpp < target_bpp - 1e-2:
                    beta.assign(beta / 1.001)

            # Linearly annealed beta
            if anneal:
                current_beta = beta * tf.minimum(1., tf.cast(ckpt.step, tf.float32) / annealing_end)
            else:
                current_beta = beta

            loss = -log_likelihood + current_beta * kld

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Once the model parameters are updated, we also update their exponential moving average.
        model.update_ema_variables()

        true_kls = model.kl_divergence(empirical=True, minimum_kl=0., reduce=False)

        return loss, reconstruction, kld, bpp, log_likelihood, current_beta, true_kls

    # Restore previous session
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        _log.info(f"Restored model from {manager.latest_checkpoint}")
        _log.info(f"Restored learning rate: {learn_rate}")

        if learn_rate > learning_rate:
            learn_rate.assign(learning_rate)
            _log.info(f"Learning rate was reassigned to: {learn_rate}")
    else:
        _log.info("Initializing model from scratch.")

    bpp_buffer = tf.zeros(bpp_buffer_size, dtype=beta.dtype)

    for batch in dataset.take(iters - int(ckpt.step)):

        # Increment the training step
        ckpt.step.assign_add(1)

        # Decrease learning rate after a while
        if int(ckpt.step) == drop_learning_rate_after_iter:
            learn_rate.assign(learn_rate * learning_rate_drop_rate)

        if int(ckpt.step) == 2 * drop_learning_rate_after_iter:
            learn_rate.assign(learn_rate * learning_rate_drop_rate)

        if int(ckpt.step) == 3 * drop_learning_rate_after_iter:
            learn_rate.assign(learn_rate * learning_rate_drop_rate)

        if int(ckpt.step) == 4 * drop_learning_rate_after_iter:
            learn_rate.assign(learn_rate * learning_rate_drop_rate)

        loss, reconstruction, kld, bpp, log_likelihood, current_beta, true_kls = train_step(batch, model, beta)

        true_kl = tf.reduce_sum(true_kls)

        if tf.math.is_nan(loss) or tf.math.is_inf(loss) or kld == 0.:
            raise Exception(f"Loss blew up: {loss:.3f}, NLL: {-log_likelihood:.3f}, KL: {kld:.3f}")

        if int(ckpt.step) % tensorboard_log_freq == 1:
            # Save model
            save_path = manager.save()
            _log.info(f"Step {int(ckpt.step)}: Saved model to {save_path}")

            true_elbo = log_likelihood - kld

            with summary_writer.as_default():
                tfs.scalar(name="Beta", data=current_beta, step=ckpt.step)
                tfs.scalar(name="Loss", data=loss, step=ckpt.step)
                tfs.scalar(name="NLL", data=-log_likelihood, step=ckpt.step)
                tfs.scalar(name="Total_KL_plus_Free_bits", data=kld, step=ckpt.step)
                tfs.scalar(name="Total_True_KL", data=true_kl, step=ckpt.step)
                tfs.scalar(name="Lossy_Bits_per_pixel",
                           data=true_kl / (num_pixels * log_2),
                           step=ckpt.step)
                tfs.scalar(name="Lossy_Bits_per_pixel_and_channel",
                           data=true_kl / (num_pixels * num_channels * log_2),
                           step=ckpt.step)
                tfs.scalar(name="Lossless_Bits_per_pixel",
                           data=-true_elbo / (num_pixels * log_2),
                           step=ckpt.step)
                tfs.scalar(name="Lossless_Bits_per_pixel_and_channel",
                           data=-true_elbo / (num_pixels * num_channels * log_2),
                           step=ckpt.step)
                tfs.scalar(name="Likelihood_Scale",
                           data=tf.math.exp(model.likelihood_log_scale),
                           step=ckpt.step)

                # If we are training for lossy compression
                if dataset_info["dataset_name"] in ["clic2019"]:
                    psnr = tf.image.psnr(reconstruction, batch + 0.5, max_val=1.0)
                    psnr = tf.reduce_mean(psnr)
                    ms_ssim = tf.image.ssim_multiscale(reconstruction, batch + 0.5, max_val=1.0)
                    ms_ssim = tf.reduce_mean(ms_ssim)

                    tfs.scalar(name="Average_PSNR",
                               data=psnr,
                               step=ckpt.step)

                    tfs.scalar(name="Average_MS-SSIM",
                               data=ms_ssim,
                               step=ckpt.step)

                tfs.image(name="Original", data=batch + 0.5, step=ckpt.step)
                tfs.image(name="Reconstruction", data=reconstruction, step=ckpt.step)

                for i, kl in enumerate(true_kls):
                    tfs.scalar(name=f"KL/dim_{i + 1}", data=tf.squeeze(kl), step=ckpt.step)


@ex.automain
def train_model(model, _log):
    dataset, num_pixels = load_dataset()

    if model == "vae":
        _log.info("Training a regular VAE!")
        train_vae(dataset=dataset)

    elif model in ["resnet_vae", "large_resnet_vae"]:
        _log.info("Training a ResNet VAE!")
        train_resnet_vae(dataset=dataset,
                         num_pixels=num_pixels)

    else:
        raise NotImplementedError
