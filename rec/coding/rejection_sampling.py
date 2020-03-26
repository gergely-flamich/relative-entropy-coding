import numpy as np
import math
import tensorflow as tf
import tensorflow_probability as tfp

from rec.coding.utils import CodingError

tfd = tfp.distributions


def get_t_p_mass(t, p, n_samples=100, oversampling=100):
    y = t.sample((n_samples * oversampling,))

    t_mass = -np.log(n_samples) + tf.zeros((n_samples * oversampling,))
    n_axes = len(y.shape)
    p_mass = -np.log(n_samples) + tf.reduce_sum(p.log_prob(y) - t.log_prob(y), axis=range(1, n_axes))

    log_ratios = t_mass - p_mass
    ind = tf.argsort(log_ratios)
    reduced_ind = tf.gather(ind, range(n_samples * oversampling)[oversampling // 2::oversampling])

    return tf.gather(log_ratios, reduced_ind), tf.gather(t_mass, reduced_ind), tf.gather(p_mass, reduced_ind)


def get_r_pstar(log_ratios, t_mass, p_mass, r_buffer_size, dtype=tf.float32):
    t_mass = tf.cast(t_mass, dtype=tf.float64)
    p_mass = tf.cast(p_mass, dtype=tf.float64)
    ratios_np = tf.exp(log_ratios).numpy()
    t_cummass_np = tf.exp(tf.math.cumulative_logsumexp(t_mass)).numpy()
    p_cummass_np = tf.exp(tf.math.cumulative_logsumexp(p_mass)).numpy()
    p_zero = float(1. - np.exp(tf.reduce_logsumexp(p_mass)))
    pstar_buffer = tf.Variable(tf.zeros((r_buffer_size,), dtype=dtype), trainable=False)
    r_buffer = tf.Variable(tf.zeros((r_buffer_size,), dtype=dtype), trainable=False)
    r = 1.
    r_buffer[0].assign(r)
    i = 1
    for r_ind, r_next in enumerate(ratios_np):
        if r_next < r:
            continue
        p_cum = p_zero + (p_cummass_np[r_ind - 1] if r_ind > 0 else 0.)
        t_cum = t_cummass_np[r_ind - 1] if r_ind > 0 else 0.
        # if final sample, the logarithm should be -infinity
        assert(r_ind != ratios_np.shape[0] - 1 or math.isclose(r_next, (1. - t_cum) / (1. - p_cum), rel_tol=1e-5))
        if r_ind == ratios_np.shape[0] - 1:
            interval = r_buffer_size - i
        else:
            interval = min(r_buffer_size - i,
                           int(math.ceil(np.log((r_next - (1. - t_cum) / (1. - p_cum)) /
                                                (r - (1. - t_cum) / (1. - p_cum))) // np.log(p_cum))))

        # Work in log for numerical stability
        r_slice = -tf.exp(np.log(p_cum) * (1. + tf.range(interval, dtype=dtype))
                          + np.log((1. - t_cum) / (1. - p_cum) - r)) + (1. - t_cum) / (1. - p_cum)
        r_buffer[i:i+interval].assign(r_slice)
        pstar_buffer[i-1:i+interval-1].assign((1. - p_cum) * r_buffer[i-1:i+interval-1] + t_cum)
        r = np.power(p_cum, interval) * (r - (1. - t_cum) / (1. - p_cum)) + (1. - t_cum) / (1. - p_cum)
        i += interval
        if i == r_buffer_size:
            pstar_buffer[r_buffer_size - 1].assign((1. - p_cum) * r + t_cum)
            break
        if r_ind == ratios_np.shape[0] - 1:
            raise CodingError('R Buffer incomplete after processing all samples. This is a bug.')
    return r_buffer, pstar_buffer


def gaussian_rejection_sample_small(t_dist,
                                    p_dist,
                                    sample_buffer_size,
                                    r_buffer_size,
                                    seed=42069):
    """
    Encodes a single sample from a Gaussian target distribution using another Gaussian coding distribution.
    Note that the runtime of this function is O(e^KL(q || p)), hence it is the job of the caller to potentially
    partition a larger Gaussian into smaller codable chunks.

    :param t_dist: the target Gaussian
    :param p_dist: the coding/proposal Gaussian
    :param sample_buffer_size: buffer size of the samples
    :param r_buffer_size: buffer size of rejection sampling, samples beyond this index are treated as if they were drawn
     at with index
    :param seed: seed that defines the infinite string of random samples from the coding distribution.
    :return: (sample, index) - tuple containing the sample and the index
    """
    assert(r_buffer_size % sample_buffer_size == 0)
    log_ratios, t_mass, p_mass = get_t_p_mass(t_dist, p_dist, n_samples=100, oversampling=100)
    r_buffer, pstar_buffer = get_r_pstar(log_ratios, t_mass, p_mass, r_buffer_size=r_buffer_size)
    kl = tf.reduce_sum(tfp.distributions.kl_divergence(t_dist, p_dist))
    if kl >= 20.:
        raise CodingError('KL divergence={} is too high for rejection sampling'.format(kl))

    tf.random.set_seed(seed)
    i = 0
    for _ in range(int(r_buffer_size // sample_buffer_size)):
        samples = p_dist.sample((sample_buffer_size,), seed=seed + i // sample_buffer_size)
        n_axes = len(samples.shape)
        sample_ratios = tf.reduce_sum(t_dist.log_prob(samples) - p_dist.log_prob(samples), axis=range(1, n_axes))
        accepted = (tf.exp(sample_ratios) - r_buffer[i:i+sample_buffer_size]) / \
                   (1. - pstar_buffer[i:i+sample_buffer_size]) + tf.random.uniform(shape=sample_ratios.shape)
        accepted_ind = tf.where(accepted > 0.)
        if accepted_ind.shape[0] > 0:
            index = int(accepted_ind[0, 0])
            return i + index, samples[index]
        i += sample_buffer_size

    # If not finished in buffer, we accept anything above ratio r
    r = r_buffer[-1]
    while True:
        samples = p_dist.sample((sample_buffer_size,), seed=seed + i // sample_buffer_size)
        sample_ratios = tf.reduce_sum(t_dist.log_prob(samples) - p_dist.log_prob(samples), axis=range(1, n_axes))
        accepted_ind = tf.where(sample_ratios > tf.math.log(r))
        if accepted_ind.shape[0] > 0:
            index = int(accepted_ind[0, 0])
            return i + index, samples[index]
        else:
            i += sample_buffer_size
