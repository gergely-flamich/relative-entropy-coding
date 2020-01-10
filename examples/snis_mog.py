from rec.core.modules.snis_distribution import SNISDistribution

import tensorflow as tf
tfl = tf.keras.layers
tfs = tf.summary

import tensorflow_probability as tfp
tfd = tfp.distributions

import numpy as np
from tqdm import tqdm
import argparse
import datetime

import matplotlib.pyplot as plt


def main(args):
    # ---------------------------------------------------------------------
    # Setup problem
    # ---------------------------------------------------------------------
    component_means = np.array([[-1, -1],
                                [-1, 0],
                                [-1, 1],
                                [0, -1],
                                [0, 0],
                                [0, 1],
                                [1, -1],
                                [1, 0],
                                [1, 1]], dtype=np.float32)
    component_stds = 0.1

    components = [tfd.Normal(loc=loc, scale=component_stds) for loc in component_means]

    mixture = tfd.Mixture(cat=tfd.Categorical(probs=[[1 / 9] * 9] * 2),
                          components=components)

    # ---------------------------------------------------------------------
    # Create dataset
    # ---------------------------------------------------------------------
    data = mixture.sample(args.num_samples)

    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.shuffle(2000)
    ds = ds.repeat()
    ds = ds.batch(128)

    # ---------------------------------------------------------------------
    # SNIS energy function to be learned
    # ---------------------------------------------------------------------
    sn = tf.keras.Sequential([
        tfl.Dense(units=args.hidden,
                  activation=tf.nn.tanh),
        tfl.Dense(units=args.hidden,
                  activation=tf.nn.tanh),
        tfl.Dense(units=1)
    ])

    # Create SNIS distribution
    sd = SNISDistribution(energy_fn=sn,
                          prior=tfd.Normal(loc=[0., 0.], scale=1.),
                          K=128)

    optimizer = tf.optimizers.Adam(3e-4)

    # ---------------------------------------------------------------------
    # Setup checkpoint
    # ---------------------------------------------------------------------

    ckpt = tf.train.Checkpoint(step=tf.Variable(1, dtype=tf.int64),
                               sd=sd,
                               optimizer=optimizer)

    manager = tf.train.CheckpointManager(ckpt, args.save_dir, max_to_keep=3)

    # Initialize the model by passing zeros through it
    sd.log_prob(tf.zeros([1, 2]))

    # Restore previous session
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print(f"Restored model from {manager.latest_checkpoint}")
    else:
        print("Initializing model from scratch.")

    # ---------------------------------------------------------------------
    # Train
    # ---------------------------------------------------------------------
    if not args.no_train:

        # Get summary writer
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = args.save_dir + "/logs/" + current_time + "/train"

        summary_writer = tfs.create_file_writer(log_dir)

        for batch in tqdm(ds.take(args.iters), total=args.iters):

            ckpt.step.assign_add(1)

            with tf.GradientTape() as tape:

                loss = -tf.reduce_mean(sd.log_prob_lower_bound(batch))

            gradients = tape.gradient(loss, sd.trainable_variables)
            optimizer.apply_gradients(zip(gradients, sd.trainable_variables))

            if int(ckpt.step) % args.log_freq == 0:
                manager.save()

                with summary_writer.as_default():
                    tfs.scalar("Log-Likelihood", -loss, step=ckpt.step)

    # ---------------------------------------------------------------------
    # Visualize results
    # ---------------------------------------------------------------------
    side = np.linspace(-2, 2, 100)
    xs, ys = np.meshgrid(side, side)

    points = np.vstack((xs.reshape([1, -1]), ys.reshape([1, -1])))
    points = tf.transpose(tf.cast(points, tf.float32))

    # Get ground truth plot
    likelihoods = tf.reduce_prod(mixture.prob(points), axis=1).numpy()

    # Get predicted plot
    un_log_probs = tf.reduce_sum(sd.prior.log_prob(points), axis=1, keepdims=True) + sd.energy_fn(points)
    log_Z = tf.reduce_logsumexp(un_log_probs)

    log_probs = un_log_probs - log_Z

    # Plot stuff
    plt.figure(figsize=(18, 5.5))
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.15)

    plt.subplot(121)
    plt.title("Ground Truth", fontsize=18)
    plt.pcolormesh(xs, ys, likelihoods.reshape(xs.shape))
    plt.axis("off")

    plt.subplot(122)
    plt.title("SNIS Density estimation", fontsize=18)
    plt.pcolormesh(xs, ys, tf.math.exp(log_probs).numpy().reshape(xs.shape))
    plt.axis("off")

    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir", "-S", required=True,
                        help="Path for the model checkpoints.")

    parser.add_argument("--no_train", action="store_true", default=False)

    parser.add_argument("--num_samples", type=int, default=1000000,
                        help="Number of samples to draw from the MoG for the dataset")
    parser.add_argument("--hidden", type=int, default=20)
    parser.add_argument("--iters", type=int, default=250000)
    parser.add_argument("--log_freq", type=int, default=1000)

    args = parser.parse_args()

    main(args)
