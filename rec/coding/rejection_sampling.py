import numpy as np
import math

import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot as plt

from rec.coding.utils import CodingError

tfd = tfp.distributions


def get_t_p_mass(t, p, n_samples=100, oversampling=100):
    y = t.sample((n_samples * oversampling,))

    t_mass = -np.log(n_samples) + tf.zeros((n_samples * oversampling,))
    p_mass = -np.log(n_samples) + tf.reduce_sum(p.log_prob(y) - t.log_prob(y), axis=1)

    log_ratios = t_mass - p_mass
    ind = tf.argsort(log_ratios)
    reduced_ind = tf.gather(ind, range(n_samples * oversampling)[oversampling // 2::oversampling])

    return tf.gather(log_ratios, reduced_ind), tf.gather(t_mass, reduced_ind), tf.gather(p_mass, reduced_ind)


# Fast, vectorized version
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
                                    R_buffer_size,
                                    seed=42069):
    assert(R_buffer_size % sample_buffer_size == 0)
    log_ratios, t_mass, p_mass = get_t_p_mass(t_dist, p_dist)
    R_buffer, pstar_buffer = get_r_pstar(log_ratios, t_mass, p_mass, r_buffer_size=R_buffer_size)
    print('Rejection sampling with KL={}'.format(tf.reduce_sum(tfp.distributions.kl_divergence(t_dist, p_dist))))
    i = 0
    for _ in range(int(R_buffer_size // sample_buffer_size)):
        samples = p_dist.sample((sample_buffer_size,), seed=seed)
        n_axes = len(samples.shape)
        sample_ratios = tf.reduce_sum(t_dist.log_prob(samples) - p_dist.log_prob(samples), axis=range(1, n_axes))
        accepted = (tf.exp(sample_ratios) - R_buffer[i:i+sample_buffer_size]) / \
                   (1. - pstar_buffer[i:i+sample_buffer_size]) + tf.random.uniform(shape=sample_ratios.shape)
        accepted_ind = tf.where(accepted > 0.)
        if accepted_ind.shape[0] > 0:
            index = int(accepted_ind[0, 0])
            return i + index, samples[index]
        i += sample_buffer_size
    # If not finished in buffer, we accept anything above ratio R
    R = R_buffer[-1]
    while True:
        samples = p_dist.sample((sample_buffer_size,), seed=seed)
        sample_ratios = tf.reduce_sum(t_dist.log_prob(samples) - p_dist.log_prob(samples), axis=range(1, n_axes))
        accepted_ind = tf.where(sample_ratios > tf.math.log(R))
        if accepted_ind.shape[0] > 0:
            index = int(accepted_ind[0, 0])
            return i + index, samples[index]
        else:
            i += sample_buffer_size


def get_aux_distribution(t, p, aux_var):
    p_var = tf.math.pow(p.scale, 2)
    t_var = tf.math.pow(t.scale, 2)
    ta_mean = (t.loc - p.loc) * aux_var / p_var
    ta_var = t_var * tf.math.pow(aux_var, 2) / tf.math.pow(p_var, 2) + aux_var * (p_var - aux_var) / p_var
    pa = tfp.distributions.Normal(loc=tf.zeros_like(ta_mean), scale=tf.sqrt(aux_var))
    ta = tfp.distributions.Normal(loc=ta_mean, scale=tf.sqrt(ta_var))
    return ta, pa


def get_conditionals(t, p, aux_var, a):
    p_var = tf.math.pow(p.scale, 2)
    t_var = tf.math.pow(t.scale, 2)
    new_t_mean = (p.loc + (a * t_var * p_var + (t.loc - p.loc) * (p_var - aux_var) * p_var) /
               (t_var * aux_var + p_var * (p_var - aux_var)))
    new_t_var = t_var * p_var * (p_var - aux_var) / (aux_var * t_var + p_var * (p_var - aux_var))
    new_p = tfp.distributions.Normal(p.loc + a, tf.sqrt(p_var - aux_var))
    new_t = tfp.distributions.Normal(new_t_mean, tf.sqrt(new_t_var))
    return new_t, new_p


def preprocessing_auxiliary_ratios(t_list, p_list, target_kl):
    image_ratio_list = []
    for t, p in zip(t_list, p_list):
        n_aux = 1 + int(tf.reduce_sum(tfp.distributions.kl_divergence(t, p)) // target_kl)
        ratio_list = []
        aux_ratio_untransformed = tf.Variable(-2., trainable=True)
        for i in range(n_aux, 1, -1):
            kl = tf.reduce_sum(tfp.distributions.kl_divergence(t, p))
            opt = tf.optimizers.SGD(learning_rate=0.001)
            def get_loss():
                aux_ratio = tf.math.sigmoid(aux_ratio_untransformed)
                aux_var = aux_ratio * tf.math.pow(p.scale, 2)
                ta, pa = get_aux_distribution(t, p, aux_var)
                aux_kl = tf.reduce_sum(tfp.distributions.kl_divergence(ta, pa))
                return tf.math.pow(aux_kl - kl / i, 2)
            for _ in range(50 if i < n_aux else 500):
                opt.minimize(get_loss, var_list=[aux_ratio_untransformed])

            aux_ratio = tf.math.sigmoid(aux_ratio_untransformed)
            aux_var = aux_ratio * tf.math.pow(p.scale, 2)
            ta, pa = get_aux_distribution(t, p, aux_var)
            aux_kl = tf.reduce_sum(tfp.distributions.kl_divergence(ta, pa))
            print('{} aux_ratio={}, KL={}, Aux KL={}, ratio={}'.format(i, aux_ratio, kl, aux_kl, kl/aux_kl))
            t, p = get_conditionals(t, p, aux_var, ta.sample())
            ratio_list.append(float(aux_ratio))
        ratio_list.reverse()
        image_ratio_list.append(ratio_list)

    average_ratios = np.zeros((max([len(l) for l in image_ratio_list]),))
    denominator = np.zeros((max([len(l) for l in image_ratio_list]),))
    for l in image_ratio_list:
        average_ratios[:len(l)] += l
        denominator[:len(l)] += 1
        plt.plot(l, c='b')
    average_ratios /= denominator
    plt.plot(average_ratios, c='r')
    plt.show()

    return average_ratios


# TODO: remove eventually
def gaussian_rejection_sample_large(t_dist,
                                    p_dist,
                                    target_kl,
                                    auxiliary_ratios,
                                    sample_buffer_size,
                                    r_buffer_size,
                                    seed=42069):
    kl = tf.reduce_sum(tfp.distributions.kl_divergence(t_dist, p_dist))
    n_aux = 1 + int(kl // target_kl)
    indicies = []
    for aux_ratio in auxiliary_ratios[:n_aux - 1][::-1]:
        print('KL={}, aux_ratio={}'.format(tf.reduce_sum(tfp.distributions.kl_divergence(t_dist, p_dist)), aux_ratio))
        aux_var = aux_ratio * tf.math.pow(p_dist.scale, 2)
        ta, pa = get_aux_distribution(t_dist, p_dist, aux_var)
        index, sample = gaussian_rejection_sample_small(ta, pa, sample_buffer_size, r_buffer_size, seed)
        print('Sample found at {}'.format(index))
        indicies.append(index)
        t_dist, p_dist = get_conditionals(t_dist, p_dist, aux_var, sample)
    index, sample = gaussian_rejection_sample_small(t_dist, p_dist, sample_buffer_size, r_buffer_size, seed)
    indicies.append(index)
    return sample, indicies


