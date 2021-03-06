import tensorflow as tf
import tensorflow_probability as tfp

from rec.coding.utils import CodingError, stateless_gumbel_sample

tfd = tfp.distributions


def encode_gaussian_importance_sample(t_loc,
                                      t_scale,
                                      p_loc,
                                      p_scale,
                                      coding_bits,
                                      seed,
                                      log_weighting_fn=None,
                                      alpha=float('inf')):
    """
    Encodes a single sample from a Gaussian target distribution using another Gaussian coding distribution.
    Note that the runtime of this function is O(e^KL(q || p)), hence it is the job of the caller to potentially
    partition a larger Gaussian into smaller codable chunks.

    :param coding_bits: number of bits to use to code each sample
    :param t_loc: location parameter of the target Gaussian
    :param t_scale: scale parameter of the target Gaussian
    :param p_loc: location parameter of the coding/proposal Gaussian
    :param p_scale: scale parameter of the coding/proposal Gaussian
    :param seed: seed that defines the infinite string of random samples from the coding distribution.
    :param alpha: draw the seed according to the L_alpha norm. alpha=1 results in sampling the atomic distribution
    defined by the importance weights, and alpha=inf just selects the sample with the maximal importance weight. Must
    be in the range [1, inf)
    :return: (sample, index) - tuple containing the sample and the
    """

    if alpha < 1.:
        raise CodingError(f"Alpha must be in the range [1, inf), but {alpha} was given!")

    # Fix seed
    tf.random.set_seed(seed)

    # Standardize the target w.r.t the coding distribution
    t_loc = (t_loc - p_loc) / p_scale
    t_scale = t_scale / p_scale

    target = tfd.Normal(loc=t_loc,
                        scale=t_scale)

    proposal = tfd.Normal(loc=tf.zeros_like(p_loc),
                          scale=tf.ones_like(p_scale))

    # We need to draw approximately e^KL samples to be guaranteed a low bias sample
    num_samples = tf.cast(tf.math.ceil(tf.exp(coding_bits * tf.math.log(2.))), tf.int32)

    # Draw e^KL samples
    samples = proposal.sample(num_samples)

    # Calculate the log-unnormalized importance_weights
    if log_weighting_fn is None:
        log_importance_weights = tf.reduce_sum(target.log_prob(samples) - proposal.log_prob(samples),
                                               axis=range(1, tf.rank(t_loc) + 1))
    else:
        log_importance_weights = log_weighting_fn(samples)

    # If we are using the infinity norm, we can just take the argmax as a shortcut
    if tf.math.is_inf(alpha):
        index = tf.argmax(log_importance_weights)

    # If we are using any other alpha, we just calculate the atomic distribution
    else:
        # Sample index using the Gumbel-max trick
        perturbed = alpha * log_importance_weights + stateless_gumbel_sample(log_importance_weights.shape, seed + 1)

        index = tf.argmax(perturbed)

    chosen_sample = samples[index, ...]

    # Rescale the sample
    chosen_sample = p_scale * chosen_sample + p_loc

    return index, chosen_sample


def decode_gaussian_importance_sample(p_loc, p_scale, index, seed):
    """
    Decodes a sample encoded using a Gaussian distribution
    :param p_loc:
    :param p_scale:
    :param index:
    :param seed:
    :return:
    """

    # Fix seed
    tf.random.set_seed(seed)

    proposal = tfd.Normal(loc=tf.zeros_like(p_loc),
                          scale=tf.ones_like(p_scale))

    # Add 1, because the codes begin at 0 not 1
    samples = proposal.sample(tf.cast(index, tf.int64) + 1)

    sample = p_scale * samples[index] + p_loc

    return sample



