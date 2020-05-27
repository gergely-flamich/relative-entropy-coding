from sacred import Experiment

import datetime

import tensorflow as tf

import tensorflow_probability as tfp

from rec.models.lossy import Large1LevelVAE, Large2LevelVAE, Large4LevelVAE
from rec.core.utils import gaussian_blur

from datasets import data_ingredient, quantize_image

tfs = tf.summary
tfd = tfp.distributions

log_2 = tf.math.log(2.)

ex = Experiment("train_lossy_model", ingredients=[data_ingredient])


@ex.config
def config(dataset):
    model_save_base_dir = "/scratch/gf332/models/relative-entropy-coding/lossy/"

    model = "large_level_1_vae"

    beta = 1.
    anneal_kl = False
    anneal_end = 60000

    optimizer = "adam"
    learning_rate = 1e-4

    iters = 3000000
    shuffle_buffer_size = 5000
    batch_size = 8
    num_prefetch = 32

    if model == "large_level_1_vae":
        num_filters = 192

        loss_fn = "mse"

        model_config = {
            "num_filters": num_filters,
        }

        model_save_dir = f"{model_save_base_dir}/{dataset['dataset_name']}/{model}/{loss_fn}/" \
                         f"beta_{beta:.3f}_filters_{num_filters}"

    elif model == "large_level_2_vae":
        level_1_filters = 196
        level_2_filters = 128

        loss_fn = "mse"

        model_config = {
            "level_1_filters": level_1_filters,
            "level_2_filters": level_2_filters
        }

        model_save_dir = f"{model_save_base_dir}/{dataset['dataset_name']}/{model}/{loss_fn}/" \
                         f"beta_{beta:.3f}_filters_{level_1_filters}_{level_2_filters}"

    elif model == "large_level_4_vae":
        level_1_filters = 196
        level_2_filters = 196
        level_3_filters = 128
        level_4_filters = 128

        loss_fn = "mse"

        model_config = {
            "level_1_filters": level_1_filters,
            "level_2_filters": level_2_filters,
            "level_3_filters": level_3_filters,
            "level_4_filters": level_4_filters,
        }

        model_save_dir = f"{model_save_base_dir}/{dataset['dataset_name']}/{model}/{loss_fn}/" \
                         f"beta_{beta:.3f}_filters_{level_1_filters}_{level_2_filters}_{level_3_filters}_{level_4_filters}"

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"{model_save_dir}/logs/{current_time}/train"

    log_freq = 300


def read_png(filename):
    """
    Loads a PNG image file. Taken from Balle's implementation
    """
    image_raw = tf.io.read_file(filename)
    image = tf.image.decode_png(image_raw, channels=3)
    image = tf.cast(image, tf.float32)
    image = image / 255.

    return image

@tf.function
def train(summary_writer,
          dataset,

          model,
          ckpt,
          manager,

          batch_size,
          iters,

          loss_fn,
          likelihood_log_scale,
          optimizer,

          beta,
          anneal_kl,
          anneal_end,

          log_freq):

    discretized_logistic = "discretized_logistic"

    print("Tracing!")
    print(likelihood_log_scale.trainable)

    for batch in dataset.take(tf.cast(iters - int(ckpt.step), tf.int64)):
        batch.set_shape([batch_size] + batch.shape[1:])

        # Increment the training step
        ckpt.step.assign_add(1)

        with tf.GradientTape() as tape:

            likelihood_scale = tf.math.exp(likelihood_log_scale)

            reconstruction = model(batch)

            distortion = -float('inf')

            mse = tf.reduce_mean(tf.math.squared_difference(batch, reconstruction))
            # Correction for the rescaling
            mse = mse * 255. ** 2.

            mae = tf.reduce_mean(tf.math.abs(batch - reconstruction))
            # Correction for the rescaling
            mae = mae * 255.

            ms_ssim = tf.reduce_mean(1. - tf.image.ssim_multiscale(batch,
                                                                   reconstruction,
                                                                   max_val=1.,
                                                                   power_factors=(1., 1., 1., 1., 1.)))
            # Pseudo-correction for the rescaling
            ms_ssim = ms_ssim * 255.

            if loss_fn == "mse":
                distortion = mse

            elif loss_fn == "ms-ssim":
                distortion = ms_ssim

            elif loss_fn == "mae":
                distortion = mae

            elif loss_fn == "mae-ms-ssim":
                alpha = 0.84

                blurred_mae = gaussian_blur(mae, kernel_size=11, sigma=8.)

                distortion = alpha * ms_ssim + (1 - alpha) * blurred_mae

            elif loss_fn == discretized_logistic:
                binsize = 1. / 256.

                clipped_reconstruction = tf.clip_by_value(reconstruction - 0.5,
                                                          -0.5 + 1. / 512.,
                                                          0.5 - 1. / 512.)
                centered_batch = batch - 0.5

                # Discretize the output
                discretized_input = tf.math.floor(centered_batch / binsize) * binsize
                # print("cr range", tf.reduce_min(clipped_reconstruction), tf.reduce_max(clipped_reconstruction))
                # print("cb range", tf.reduce_min(centered_batch), tf.reduce_max(centered_batch))
                # print("max diff", tf.reduce_max(tf.abs((discretized_input - clipped_reconstruction))))

                discretized_input = (discretized_input - clipped_reconstruction) / likelihood_scale

                log_likelihood = tf.nn.sigmoid(discretized_input + binsize / likelihood_scale)
                log_likelihood = log_likelihood - tf.nn.sigmoid(discretized_input)

                log_likelihood = tf.math.log(log_likelihood + 1e-7)

                # distortion is average distortion per image!
                distortion = -tf.reduce_mean(tf.reduce_sum(log_likelihood, axis=3)) / log_2

            else:
                raise NotImplementedError

            klds = model.kl_divergence()

            kld = tf.reduce_sum(klds)

            # Divide by the number of pixels and switch from nats to bits
            bpp = kld / (batch.shape[1] * batch.shape[2] * log_2)

            if anneal_kl:
                # Starts beta at a high value, and anneals it linearly to the desired value
                kl_coeff = tf.minimum(1., tf.cast(ckpt.step, tf.float32) / anneal_end)

            else:
                kl_coeff = 1.

            loss = beta * distortion + kl_coeff * bpp

            # print(loss)
            # print(distortion)
            # print(kld)
            # print(likelihood_scale)
            # print()

        stop = False
        for var in model.trainable_variables:
            if tf.reduce_any(tf.math.is_nan(var)) or tf.reduce_any(tf.math.is_inf(var)):
                tf.print(var.name, "was messed up", var)
                stop = True

        if stop:
            tf.print("stopping")
            break

        gradients = tape.gradient(loss, model.trainable_variables + [likelihood_log_scale])
        #gradients = tf.clip_by_global_norm(gradients, clip_norm=20.)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables + [likelihood_log_scale]))

        if tf.math.is_nan(loss) or tf.math.is_inf(loss) or kld == 0.:
            tf.print("Loss blew up:", loss, "Distortion:", distortion, "KL:", kld, "Quitting.")
            tf.print("Image", tf.reduce_any(tf.math.is_nan(batch)), tf.reduce_any(tf.math.is_inf(batch)))
            tf.print("Reconstruction", tf.reduce_any(tf.math.is_nan(reconstruction)), tf.reduce_any(tf.math.is_inf(reconstruction)))

            bad_indices = tf.where(tf.math.is_nan(reconstruction))
            tf.print(bad_indices)

            tf.print(tf.gather_nd(batch, bad_indices))

            break

        if int(ckpt.step) % log_freq == 0:

            # CheckpointManager currently cannot save in graph mode (using tf.function)
            # Solution from: https://github.com/tensorflow/tensorflow/issues/28523#issuecomment-493194661
            save_path = tf.py_function(manager.save, [], [tf.string])
            tf.print(f"Step:", int(ckpt.step), "Saved model to", save_path[0])

            psnr = tf.reduce_mean(tf.image.psnr(reconstruction, batch, max_val=1.0))
            ms_ssim_actual = tf.reduce_mean(tf.image.ssim_multiscale(reconstruction, batch, max_val=1.0))
            ms_ssim_actual_db = -10. * tf.math.log(1. - ms_ssim_actual) / tf.math.log(10.)

            with summary_writer.as_default():

                tfs.scalar(name="KL_Coeff", data=kl_coeff, step=ckpt.step)
                tfs.scalar(name="Loss", data=loss, step=ckpt.step)
                tfs.scalar(name="Likelihood_scale", data=likelihood_scale, step=ckpt.step)
                tfs.scalar(name="MSE", data=mse, step=ckpt.step)
                tfs.scalar(name="MAE", data=mae, step=ckpt.step)
                tfs.scalar(name="MS-SSIM_(distortion)", data=ms_ssim, step=ckpt.step)
                tfs.scalar(name="MS-SSIM", data=ms_ssim_actual, step=ckpt.step)
                tfs.scalar(name="MS-SSIM_(dB)", data=ms_ssim_actual_db, step=ckpt.step)
                tfs.scalar(name="PSNR", data=psnr, step=ckpt.step)
                tfs.scalar(name="Distortion", data=distortion, step=ckpt.step)
                tfs.scalar(name="Total_KL", data=kld, step=ckpt.step)
                tfs.scalar(name="BPP",
                           data=bpp,
                           step=ckpt.step)
                tfs.image(name="Original", data=quantize_image(batch), step=ckpt.step)
                tfs.image(name="Reconstruction", data=quantize_image(reconstruction), step=ckpt.step)

                for i, kl in enumerate(klds):
                    tfs.scalar(name=f"KL/dim_{i + 1}", data=tf.squeeze(kl), step=ckpt.step)

                summary_writer.flush()


@ex.automain
def run(dataset,

        model,
        model_config,

        batch_size,
        shuffle_buffer_size,
        num_prefetch,
        iters,

        loss_fn,
        optimizer,
        learning_rate,

        beta,
        anneal_kl,
        anneal_end,

        model_save_dir,
        log_dir,
        log_freq,

        _seed):

    tf.random.set_seed(_seed)

    # -------------------------------------------------------------------------
    # Prepare the dataset
    # -------------------------------------------------------------------------
    dataset_path = dataset["dataset_path"]

    with tf.device("/CPU:0"):

        dataset = tf.data.Dataset.list_files(f"{dataset_path}/train/*.png")
        dataset = dataset.shuffle(shuffle_buffer_size, seed=_seed, reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.map(read_png, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(lambda x: tf.image.random_crop(x, (256, 256, 3)))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(num_prefetch)

    # -------------------------------------------------------------------------
    # Create model and optimizer
    # -------------------------------------------------------------------------
    model = {
        "large_level_1_vae": Large1LevelVAE,
        "large_level_2_vae": Large2LevelVAE,
        "large_level_4_vae": Large4LevelVAE
    }[model](**model_config)

    # Initialize model
    model.build(input_shape=(batch_size, 256, 256, 3))

    learn_rate = learning_rate
    #learn_rate = tf.Variable(learning_rate, trainable=False)
    beta = tf.Variable(beta, dtype=tf.float32, name="beta", trainable=False)

    optimizer = {
        "adam": tf.optimizers.Adam,
        "adamax": tf.optimizers.Adamax
    }[optimizer](learn_rate)

    likelihood_log_scale = tf.Variable(0.,
                                   name="likelihood_log_scale",
                                   trainable=loss_fn == "discretized_logistic")

    # -------------------------------------------------------------------------
    # Create Checkpoints
    # -------------------------------------------------------------------------
    optim_step = tf.Variable(1, dtype=tf.int64, name="optim_step", trainable=False)

    ckpt = tf.train.Checkpoint(step=optim_step,
                               model=model,
                               #learn_rate=learn_rate,
                               likelihood_log_scale=likelihood_log_scale,
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

    # Restore previous session
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        tf.print(f"Restored model from {manager.latest_checkpoint}")

    else:
        tf.print("Initializing model from scratch.")

    train(summary_writer=summary_writer,
          dataset=dataset,

          model=model,
          ckpt=ckpt,
          manager=manager,

          batch_size=batch_size,
          iters=iters,

          loss_fn=loss_fn,
          likelihood_log_scale=likelihood_log_scale,
          optimizer=optimizer,

          beta=beta,
          anneal_kl=anneal_kl,
          anneal_end=anneal_end,

          log_freq=log_freq)
