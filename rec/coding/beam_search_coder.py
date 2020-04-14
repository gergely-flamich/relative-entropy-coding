import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np

from rec.coding.utils import CodingError
from rec.coding.coder import GaussianCoder, get_auxiliary_target, get_auxiliary_coder, get_conditional_target, \
    get_conditional_coder

tfl = tf.keras.layers
tfd = tfp.distributions


def sample_with_seed(dist, shape, seed):
    tf.random.set_seed(seed)
    return dist.sample(shape, seed=seed)


class BeamSearchCoder(GaussianCoder):
    def __init__(self,
                 kl_per_partition,
                 n_beams,
                 extra_samples=1.,
                 name="gaussian_encoder",
                 **kwargs):

        super().__init__(name=name, kl_per_partition=kl_per_partition, sampler=None, **kwargs)
        self.n_beams = n_beams
        self.n_samples = np.exp(kl_per_partition * extra_samples)
        assert(self.n_samples > self.n_beams)

    def encode_block(self, target_dist, coding_dist, seed, update_sampler=False):
        if not self._initialized:
            raise CodingError("Coder has not been initialized yet, please call update_auxiliary_variance_ratios() first!")

        if target_dist.loc.shape[0] != 1:
            raise CodingError("For encoding, batch size must be 1.")

        total_kl = tf.reduce_sum(tfd.kl_divergence(target_dist, coding_dist))
        print('Encoding latent variable with KL={}'.format(total_kl))
        num_aux_variables = tf.cast(tf.math.ceil(total_kl / self.kl_per_partition), tf.int32)

        # If there are more auxiliary variables needed than what we are already storing, we update our estimates
        current_max = tf.shape(self.aux_variable_variance_ratios)[0]
        if num_aux_variables > current_max:
            self.update_auxiliary_variance_ratios(target_dist, coding_dist)
            # raise CodingError("KL divergence higher than auxiliary variables can account for. "
            #                   "Update auxiliary variable ratios with high-enough KL divergence."
            #                   "Maximum possible KL divergence is {}.".format(current_max.numpy() * self.kl_per_partition))

        # We iterate backward until the second entry in ratios. The first entry is 1.,
        # in which case we just draw the final sample.
        n_dims = len(target_dist.loc.shape)
        cumulative_auxiliary_variance = 0.
        iteration = 0
        for i in range(num_aux_variables - 1, -1, -1):
            aux_variable_variance_ratio = self.aux_variable_variance_ratios[i]
            auxiliary_var = aux_variable_variance_ratio * (tf.math.pow(coding_dist.scale, 2)
                                                           - cumulative_auxiliary_variance)

            auxiliary_coder = get_auxiliary_coder(coder=coding_dist,
                                                  auxiliary_var=auxiliary_var)
            cumulative_auxiliary_coder = get_auxiliary_coder(coder=coding_dist,
                                                             auxiliary_var=auxiliary_var + cumulative_auxiliary_variance)
            auxiliary_target = get_auxiliary_target(target=target_dist,
                                                    coder=coding_dist,
                                                    auxiliary_var=auxiliary_var + cumulative_auxiliary_variance)

            if iteration > 0:
                samples = tf.stack([sample_with_seed(auxiliary_coder, (self.n_samples, ), seed=seed) for seed in
                                    seed + tf.reduce_sum(beam_indices, axis=1)], axis=1)
                combined_samples = beams + samples
                log_probs = tf.reduce_sum(auxiliary_target.log_prob(combined_samples)
                                          - cumulative_auxiliary_coder.log_prob(combined_samples),
                                          axis=range(2, n_dims + 2))
                flat_log_probs = tf.reshape(log_probs, [-1])
                sorted_ind_1d = tf.argsort(flat_log_probs, direction='DESCENDING')
                best_ind_beam = sorted_ind_1d[:self.n_beams] % self.n_beams
                best_ind_aux = sorted_ind_1d[:self.n_beams] // self.n_beams
                assert(log_probs[best_ind_aux[0], best_ind_beam[0]] == flat_log_probs[sorted_ind_1d[0]])

                beam_ind = tf.stack((best_ind_aux, best_ind_beam), axis=1)
                beams = tf.gather_nd(combined_samples, beam_ind)
                beam_indices = tf.concat((tf.gather_nd(beam_indices[:, :iteration], best_ind_beam[:, None]),
                                          best_ind_aux[:, None]), axis=1)
            else:
                samples = sample_with_seed(auxiliary_coder, (self.n_samples,), seed=seed)
                log_probs = tf.reduce_sum(auxiliary_target.log_prob(samples)
                                          - cumulative_auxiliary_coder.log_prob(samples),
                                          axis=range(1, n_dims + 1))
                sorted_ind = tf.argsort(log_probs, direction='DESCENDING')
                beams = tf.gather_nd(samples, sorted_ind[:self.n_beams, None])
                beam_indices = sorted_ind[:self.n_beams, None]

            iteration += 1
            cumulative_auxiliary_variance += auxiliary_var

        target_sample = target_dist.sample()
        target_entropy = tf.reduce_sum(target_dist.log_prob(target_sample) - coding_dist.log_prob(target_sample))
        print('Target entropy={}, log_density={}'.format(target_entropy,
                                                         tf.reduce_sum(
                                                             target_dist.log_prob(beams[0] + coding_dist.loc) -
                                                             coding_dist.log_prob(beams[0] + coding_dist.loc))))
        return list(beam_indices[0, :]), beams[0] + coding_dist.loc

    def decode_block(self, coding_dist, indices, seed):
        num_aux_variables = len(indices)

        indices.reverse()
        sample = tf.zeros_like(coding_dist.loc)
        cumulative_auxiliary_variance = 0.
        for i in range(num_aux_variables - 1, -1, -1):
            aux_variable_variance_ratio = self.aux_variable_variance_ratios[i]
            auxiliary_var = aux_variable_variance_ratio * (tf.math.pow(coding_dist.scale, 2)
                                                           - cumulative_auxiliary_variance)

            auxiliary_coder = get_auxiliary_coder(coder=coding_dist,
                                                  auxiliary_var=auxiliary_var)

            aux_samples = sample_with_seed(auxiliary_coder, (self.n_samples, ), seed=seed)
            sample = sample + aux_samples[indices[i]]
            seed += indices[i]

            cumulative_auxiliary_variance += auxiliary_var

        return sample + coding_dist.loc

    def get_codelength(self, indicies):
        return len(indicies) * np.log(self.n_samples)
