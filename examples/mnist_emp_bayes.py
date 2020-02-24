from rec.models.mnist_vae import MNISTVAE, MNISTVampVAE
from rec.core.modules.snis_distribution import SNISDistribution
from rec.core.utils import setup_logger

import datetime
import argparse
import logging

import tensorflow as tf
tfs = tf.summary
tfl = tf.keras.layers

import tensorflow_probability as tfp
tfd = tfp.distributions

import tensorflow_datasets as tfds

AVAILABLE_MODELS = [
    "gaussian",
    "mog",
    "vamp",
    "snis"
]

logger = setup_logger(__name__, level=logging.DEBUG, to_console=False, log_file=f"../logs/snis_mnist.log")


def main(args):

    logger.info("========================================")
    logger.info(f"Training a VAE with {args.model} prior.")
    logger.info("========================================")

    logger.info(f"Tensorflow was built with CUDA: {tf.test.is_built_with_cuda()}")
    logger.info(f"Tensorflow is using GPU: {tf.test.is_gpu_available()}")

    batch_size = 128
    log_freq = 1000
    anneal_end = 100000
    drop_learning_rate_after_iter = 1000000

    # Get dataset
    dataset = tfds.load("binarized_mnist",
                        data_dir=args.dataset)

    train_ds = dataset["train"]

    # Normalize data
    train_ds = train_ds.map(lambda x: tf.cast(x["image"], tf.float32))

    # Prepare the dataset for training
    train_ds = train_ds.shuffle(5000)
    train_ds = train_ds.repeat()
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(32)

    # Get model
    if args.model == "gaussian":
        model = MNISTVAE(name="gaussian_mnist_vae",
                         prior=tfd.Normal(loc=tf.zeros(args.latent_dim),
                                          scale=tf.ones(args.latent_dim)))

    elif args.model == "mog":

        num_components = 100

        loc = tf.Variable(tf.random.uniform(shape=(args.latent_dim, num_components), minval=-1., maxval=1.))
        log_scale = tf.Variable(tf.random.uniform(shape=(args.latent_dim, num_components), minval=-1., maxval=1.))

        scale = 1e-5 + tf.nn.softplus(log_scale)

        components = tfd.Normal(loc=loc,
                                scale=scale)

        mixture = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=[1. / num_components] * num_components),
                                        components_distribution=components)

        model = MNISTVAE(name="mog_mnist_vae",
                         prior=mixture)

    elif args.model == "vamp":

        model = MNISTVampVAE(name="vamp_mnist_vae",
                             latents=args.latent_dim)

    elif args.model == "snis":

        snis_network = tf.keras.Sequential([
            tfl.Dense(units=100,
                      activation=tf.nn.tanh),
            tfl.Dense(units=100,
                      activation=tf.nn.tanh),
            tfl.Dense(units=1)
        ])

        prior = SNISDistribution(energy_fn=snis_network,
                                 prior=tfd.Normal(loc=tf.zeros(args.latent_dim),
                                                  scale=tf.ones(args.latent_dim)),
                                 K=1024)

        model = MNISTVAE(name="snis_mnist_vae",
                         prior=prior)

    # Get optimizer
    learn_rate = tf.Variable(3e-4)
    optimizer = tf.optimizers.Adam(learning_rate=learn_rate)

    # Get checkpoint
    ckpt = tf.train.Checkpoint(step=tf.Variable(1, dtype=tf.int64),
                               learn_rate=learn_rate,
                               model=model,
                               optimizer=optimizer)

    manager = tf.train.CheckpointManager(ckpt, args.save_dir, max_to_keep=3)

    # Get summary writer
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = args.save_dir + "/logs/" + current_time + "/train"

    summary_writer = tfs.create_file_writer(log_dir)

    # Initialize the model by passing zeros through it
    model(tf.zeros([1, 28, 28, 1]))

    # Restore previous session
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        logger.info(f"Restored model from {manager.latest_checkpoint}")
    else:
        logger.info("Initializing model from scratch.")

    num_loss_was_nan = 0

    tau = 100
    tau = tf.cast(tau, dtype=tf.float32)

    for batch in train_ds.take(args.iters - int(ckpt.step)):

        # Increment the training step
        ckpt.step.assign_add(1)

        # Decrease learning rate after a while
        if int(ckpt.step) == drop_learning_rate_after_iter:
            learn_rate.assign(1e-5)

        with tf.GradientTape() as tape:

            reconstruction = model(batch, training=True)

            log_prob = model.likelihood.log_prob(batch)

            # Get the empirical log-likelihood per image
            nll = -tf.reduce_mean(tf.reduce_sum(log_prob, axis=[1, 2]))

            # Get the empirical KL per latent code
            kl_divergence = tf.reduce_mean(tf.reduce_sum(model.kl_divergence, axis=1))

            # Soft relaxation of the maximum KL
            soft_max_kl_divergence = 1 / tau * tf.reduce_mean(tf.reduce_logsumexp(model.kl_divergence * tau, axis=1))

            beta = tf.minimum(1., tf.cast(ckpt.step / anneal_end, tf.float32))

            gamma = tf.minimum(args.latent_dim, tf.maximum(0., tf.cast(ckpt.step / anneal_end - 0.8, tf.float32)))

            loss = nll + (beta - gamma / args.latent_dim) * kl_divergence + gamma * soft_max_kl_divergence

        if tf.reduce_min(-log_prob) > 10:
            logger.info("Mispredicted pixel!")

        # Check for NaN loss
        if tf.math.is_nan(loss):
            num_loss_was_nan += 1

            if num_loss_was_nan > 50:
                logger.error(f"Loss was NaN, stopping! nll: {nll}, KL: {kl_divergence} ")
                break
            else:
                logger.error(f"Loss was NaN, continuing for now {num_loss_was_nan}/10! nll: {nll}, KL: {kl_divergence} ")
                continue

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if int(ckpt.step) % log_freq == 0:

            # Get empirical posterior and prior log likelihoods for summaries
            post_log_lik = tf.reduce_mean(tf.reduce_sum(model.post_log_liks, axis=1))
            prior_log_lik = tf.reduce_mean(tf.reduce_sum(model.prior_log_liks, axis=1))

            expected_max_kl = tf.reduce_mean(tf.reduce_max(model.kl_divergence, axis=1))

            # Save model
            save_path = manager.save()
            logger.info(f"Step {int(ckpt.step)}: Saved model to {save_path}")

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
                tfs.scalar(name="Max_Soft_Max_KL_Diff", data=expected_max_kl - soft_max_kl_divergence, step=ckpt.step)
                tfs.scalar(name="Average_Max_KL_Diff", data=kl_divergence / args.latent_dim - expected_max_kl,
                           step=ckpt.step)
                tfs.scalar(name="Gamma", data=gamma, step=ckpt.step)

                # If we are using the VampPrior VAE, then log the evolution of the inducing points
                if args.model == "vamp":
                    tfs.image(name="Inducing Points", data=model.inducing_points, step=ckpt.step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", "-D", type=str, required=True,
                        help="Path to the binarized MNIST dataset. "
                             "If does not exist, it will be downloaded to that location.")

    parser.add_argument("--model", "-M", choices=AVAILABLE_MODELS, required=True,
                        help="Select which model to train.")

    parser.add_argument("--save_dir", "-S", required=True,
                        help="Path for the model checkpoints.")

    parser.add_argument("--iters", type=int, default=10000000)
    parser.add_argument("--latent_dim", type=int, default=50)

    args = parser.parse_args()

    main(args)
