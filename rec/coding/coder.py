import abc

import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np

from tqdm import trange

from rec.coding.utils import CodingError
from rec.coding.samplers import Sampler, RejectionSampler, ImportanceSampler

tfl = tf.keras.layers
tfd = tfp.distributions


def sigmoid_inverse(x):
    if tf.reduce_any(x < 0.) or tf.reduce_any(x > 1.):
        raise ValueError(f"x = {x} was not in the sigmoid function's range ([0, 1])!")
    x = tf.clip_by_value(x, 1e-10, 1 - 1e-10)

    return tf.math.log(x) - tf.math.log(1. - x)


class Coder(tfl.Layer, abc.ABC):

    def __init__(self,
                 name="encoder",
                 **kwargs):
        super().__init__(name=name,
                         **kwargs)


def get_auxiliary_coder(coder, auxiliary_var):
    auxiliary_coder = tfd.Normal(loc=tf.zeros_like(coder.loc), scale=tf.sqrt(auxiliary_var))

    return auxiliary_coder


def get_auxiliary_target(target, coder, auxiliary_var):
    coder_var = tf.math.pow(coder.scale, 2)
    target_var = tf.math.pow(target.scale, 2)
    auxiliary_target_mean = (target.loc - coder.loc) * auxiliary_var / coder_var
    auxiliary_target_var = target_var * tf.math.pow(auxiliary_var, 2) / tf.math.pow(coder_var, 2) \
                        + auxiliary_var * (coder_var - auxiliary_var) / coder_var
    ta = tfp.distributions.Normal(loc=auxiliary_target_mean, scale=tf.sqrt(auxiliary_target_var))
    return ta


def get_conditional_coder(coder, auxiliary_var, auxiliary_sample):
    coder_var = tf.math.pow(coder.scale, 2)

    return tfp.distributions.Normal(coder.loc + auxiliary_sample, tf.sqrt(coder_var - auxiliary_var))


def get_conditional_target(target, coder, auxiliary_var, auxiliary_sample):
    coder_var = tf.math.pow(coder.scale, 2)
    target_var = tf.math.pow(target.scale, 2)
    new_t_mean = (coder.loc + (auxiliary_sample * target_var * coder_var +
                               (target.loc - coder.loc) * (coder_var - auxiliary_var) * coder_var) /
                  (target_var * auxiliary_var + coder_var * (coder_var - auxiliary_var)))
    new_t_var = target_var * coder_var * (coder_var - auxiliary_var) / \
        (auxiliary_var * target_var + coder_var * (coder_var - auxiliary_var))
    return tfp.distributions.Normal(new_t_mean, tf.sqrt(new_t_var))


class GaussianCoder(Coder):
    def __init__(self,
                 kl_per_partition,
                 sampler: Sampler,
                 name="gaussian_encoder",
                 **kwargs):

        super().__init__(name=name,
                         **kwargs)

        # ---------------------------------------------------------------------
        # Assign parameters
        # ---------------------------------------------------------------------
        self.sampler = sampler

        self.kl_per_partition = tf.cast(kl_per_partition, tf.float32)

        # ---------------------------------------------------------------------
        # Create parameters for the auxiliary variables
        # ---------------------------------------------------------------------

        # The auxiliary variables are always scaled w.r.t the coding distribution, i.e.
        # Var[A_i] = R_i * Var_{Z_i ~ P(Z_i)}[Z_i]
        # The variance ratio at index i creates a chunk that has KL divergence 1/(i+1) times the overall KL divergence
        self.aux_variable_variance_ratios = tf.Variable(tf.constant([1.], dtype=tf.float32),
                                                        shape=tf.TensorShape([None]),
                                                        name="sum_averaged_variance_ratios",
                                                        trainable=False)

        # Counts over how many batch elements we averaged over
        self.average_counts = tf.Variable(tf.constant([1.], dtype=tf.float32),
                                          shape=tf.TensorShape([None]),
                                          name="average_counts",
                                          trainable=False)

        self._initialized = tf.Variable(False,
                                        name="coder_initialized",
                                        trainable=False)

    def update_auxiliary_variance_ratios(self,
                                         target_dist,
                                         coding_dist,
                                         relative_tolerance=1e-4,
                                         max_iters=10000,
                                         learning_rate=0.001):
        print(f"Updating {self.name}!")
        # Gather distribution statistics
        target_loc = target_dist.loc
        target_scale = target_dist.scale

        coding_loc = coding_dist.loc
        coding_scale = coding_dist.scale

        data_dims = list(range(1, tf.rank(target_loc)))

        # The first dimension is the "batch" dimension, so we preserve it
        total_kl = tf.reduce_sum(tfd.kl_divergence(target_dist, coding_dist), axis=data_dims)
        # Calculate the number of required auxiliary variables for each batch element
        num_aux_variables = 1 + tf.cast(tf.math.floor(total_kl / self.kl_per_partition), tf.int32)
        max_num_variables = tf.reduce_max(num_aux_variables)
        # get dynamic shape
        current_max = tf.shape(self.aux_variable_variance_ratios)[0]

        if max_num_variables > current_max:
            aux_variable_variance_ratios_copy = tf.identity(self.aux_variable_variance_ratios)
            self.aux_variable_variance_ratios = tf.Variable(tf.zeros((max_num_variables,), dtype=tf.float32),
                                                            shape=tf.TensorShape([None]),
                                                            name="sum_averaged_variance_ratios",
                                                            trainable=False)
            self.aux_variable_variance_ratios[:current_max].assign(aux_variable_variance_ratios_copy)

            average_counts_copy = tf.identity(self.average_counts)
            self.average_counts = tf.Variable(tf.zeros((max_num_variables,), dtype=tf.float32),
                                              shape=tf.TensorShape([None]),
                                              name="average_counts",
                                              trainable=False)
            self.average_counts[:current_max].assign(average_counts_copy)

        # Perform initialization for each possible KL ratio
        for ratio in range(max_num_variables, 1, -1):

            # We will only update the distributions with high enough KL
            indices = tf.where(num_aux_variables >= ratio)

            # Number of current batch elements
            num_elements = indices.shape[0]

            # Create dummy distributions
            target = tfd.Normal(loc=tf.gather_nd(target_loc, indices),
                                scale=tf.gather_nd(target_scale, indices))

            coder = tfd.Normal(loc=tf.gather_nd(coding_loc, indices),
                               scale=tf.gather_nd(coding_scale, indices))

            # Update KL
            total_kl = tf.reduce_sum(tfd.kl_divergence(target, coder), axis=data_dims)

            # Initialize ratio parameters
            if self.aux_variable_variance_ratios[ratio - 1] > 0.:
                init = self.aux_variable_variance_ratios[ratio - 1]
            elif ratio < max_num_variables:
                init = self.aux_variable_variance_ratios[ratio]
            else:
                init = 1. / ratio

            init = sigmoid_inverse(init)
            reparameterized_aux_variable_var_ratio = tf.Variable(init)

            # Compensate in the learning rate for the increased loss
            optimizer = tf.optimizers.SGD(learning_rate=learning_rate)

            # Optimize the current ratio using SGD
            prev_loss = np.inf

            with trange(max_iters) as progress_bar:
                for _ in progress_bar:
                    with tf.GradientTape() as tape:
                        aux_variable_variance_ratio = tf.nn.sigmoid(reparameterized_aux_variable_var_ratio)

                        auxiliary_variance = aux_variable_variance_ratio * tf.math.pow(coder.scale, 2)
                        aux_target = get_auxiliary_target(target=target,
                                                          coder=coder,
                                                          auxiliary_var=auxiliary_variance)

                        aux_coder = get_auxiliary_coder(coder=coder,
                                                        auxiliary_var=auxiliary_variance)

                        # Get the KL between q(A_i | Z) and p(A_i | Z)
                        auxiliary_kl = tf.reduce_sum(tfd.kl_divergence(aux_target, aux_coder),
                                                     axis=data_dims)

                        # Make a quadratic loss
                        # kl_loss = tf.reduce_mean(tf.math.pow(auxiliary_kl - total_kl / tf.cast(ratio, tf.float32), 2))
                        # kl_loss = tf.reduce_mean(tf.math.pow(total_kl - auxiliary_kl - self.kl_per_partition * (ratio - 1), 2))
                        aux_kl_loss = tf.where(auxiliary_kl > self.kl_per_partition,
                                               tf.math.pow(auxiliary_kl - self.kl_per_partition, 2),
                                               0.)
                        remaining_kl_loss = tf.where(total_kl - auxiliary_kl > self.kl_per_partition * (ratio - 1),
                                                     tf.math.pow((total_kl - auxiliary_kl) - self.kl_per_partition * (ratio - 1), 2),
                                                     0.)
                        kl_loss = tf.reduce_mean(aux_kl_loss + remaining_kl_loss)

                    gradient = tape.gradient(kl_loss, reparameterized_aux_variable_var_ratio)
                    optimizer.apply_gradients([(gradient, reparameterized_aux_variable_var_ratio)])

                    # Early stop if the loss decreases less than the tolerance
                    if tf.abs(prev_loss - kl_loss) < relative_tolerance:
                        break

                    prev_loss = kl_loss

                    progress_bar.set_description(f"Ratio {ratio}, {num_elements}/{target_loc.shape[0]} items - "
                                                 f"ratio: {aux_variable_variance_ratio:.4f}, "
                                                 f"avg_aux_kl: {tf.reduce_mean(auxiliary_kl):.3f}+-{tf.math.reduce_std(auxiliary_kl):.3f}, "
                                                 f"avg target_kl: {tf.reduce_mean(total_kl) / ratio:.3f}, "
                                                 f"avg kl: {tf.reduce_mean(total_kl):.3f}, "
                                                 f"loss: {kl_loss:.3f}")

            self.aux_variable_variance_ratios[ratio - 1].assign(
                (self.aux_variable_variance_ratios[ratio - 1] * self.average_counts[ratio - 1] +
                 aux_variable_variance_ratio * num_elements) /
                (self.average_counts[ratio - 1] + num_elements))
            self.average_counts[ratio - 1].assign(self.average_counts[ratio - 1] + num_elements)
            auxiliary_variance = self.aux_variable_variance_ratios[ratio - 1] * tf.math.pow(coder.scale, 2)

            # Once the optimization is finished, calculate the new target and coding distributions
            auxiliary_sample = aux_target.sample()
            target = get_conditional_target(target=target,
                                            coder=coder,
                                            auxiliary_var=auxiliary_variance,
                                            auxiliary_sample=auxiliary_sample)

            coder = get_conditional_coder(coder=coder,
                                          auxiliary_var=auxiliary_variance,
                                          auxiliary_sample=auxiliary_sample)

            # Update the distribution statistics tensors
            target_loc = tf.tensor_scatter_nd_update(target_loc, indices, target.loc)
            target_scale = tf.tensor_scatter_nd_update(target_scale, indices, target.scale)

            coding_loc = tf.tensor_scatter_nd_update(coding_loc, indices, coder.loc)
            coding_scale = tf.tensor_scatter_nd_update(coding_scale, indices, coder.scale)

        self._initialized.assign(True)

    def encode(self, target_dist, coding_dist, seed, update_sampler=False):

        if not self._initialized:
            raise CodingError("Coder has not been initialized yet, please call update_auxiliary_variance_ratios() first!")

        if target_dist.loc.shape[0] != 1:
            raise CodingError("For encoding, batch size must be 1.")

        if isinstance(self.sampler, ImportanceSampler):
            self.sampler.reset_codelength()

        indices = []

        total_kl = tf.reduce_sum(tfd.kl_divergence(target_dist, coding_dist))
        print('Encoding latent variable with KL={}'.format(total_kl))
        num_aux_variables = tf.cast(tf.math.ceil(total_kl / self.kl_per_partition), tf.int32)

        # If there are more auxiliary variables needed than what we are already storing, we update our estimates
        current_max = tf.shape(self.aux_variable_variance_ratios)[0]
        if num_aux_variables > current_max:
            raise CodingError("KL divergence higher than auxiliary variables can account for. "
                              "Update auxiliary variable ratios with high-enough KL divergence."
                              "Maximum possible KL divergence is {}.".format(current_max.numpy() * self.kl_per_partition))

        # We iterate backward until the second entry in ratios. The first entry is 1.,
        # in which case we just draw the final sample.
        for i in range(num_aux_variables - 1, 0, -1):
            aux_variable_variance_ratio = self.aux_variable_variance_ratios[i]
            auxiliary_var = aux_variable_variance_ratio * tf.math.pow(coding_dist.scale, 2)

            auxiliary_target = get_auxiliary_target(target=target_dist,
                                                    coder=coding_dist,
                                                    auxiliary_var=auxiliary_var)

            auxiliary_coder = get_auxiliary_coder(coder=coding_dist,
                                                  auxiliary_var=auxiliary_var)

            if isinstance(self.sampler, ImportanceSampler):
                self.sampler.increase_codelength(
                    tf.math.ceil(tf.reduce_sum(tfd.kl_divergence(auxiliary_target,
                                                                 auxiliary_coder)))
                )

            if update_sampler:
                self.sampler.update(auxiliary_target, auxiliary_coder)
                auxiliary_sample = auxiliary_target.sample()
                print('Sampler updated')
            else:
                index, auxiliary_sample = self.sampler.coded_sample(target=auxiliary_target,
                                                                    coder=auxiliary_coder,
                                                                    seed=seed)
                print(f'Auxiliary sample {i} found at index {index}')
                indices.append(index)
            seed += 1

            target_dist = get_conditional_target(target=target_dist,
                                                 coder=coding_dist,
                                                 auxiliary_var=auxiliary_var,
                                                 auxiliary_sample=auxiliary_sample)

            coding_dist = get_conditional_coder(coder=coding_dist,
                                                auxiliary_var=auxiliary_var,
                                                auxiliary_sample=auxiliary_sample)

        # Sample the last auxiliary variable
        if update_sampler:
            self.sampler.update(target_dist, coding_dist)
            sample = target_dist.sample()
            print('Sampler updated')
        else:
            index, sample = self.sampler.coded_sample(target=target_dist,
                                                      coder=coding_dist,
                                                      seed=seed)
            print('Auxiliary sample found at index {}'.format(index))
            indices.append(index)

        return indices, sample

    def decode(self, coding_dist, indices, seed):
        num_aux_variables = len(indices)

        indices.reverse()
        for i in range(num_aux_variables - 1, 0, -1):
            aux_variable_variance_ratio = self.aux_variable_variance_ratios[i]
            auxiliary_var = aux_variable_variance_ratio * tf.math.pow(coding_dist.scale, 2)

            auxiliary_coder = get_auxiliary_coder(coder=coding_dist,
                                                  auxiliary_var=auxiliary_var)

            auxiliary_sample = self.sampler.decode_sample(coder=auxiliary_coder,
                                                          sample_index=indices[i],
                                                          seed=seed)
            seed += 1

            coding_dist = get_conditional_coder(coder=coding_dist,
                                                auxiliary_var=auxiliary_var,
                                                auxiliary_sample=auxiliary_sample)

        sample = self.sampler.decode_sample(coder=coding_dist,
                                            sample_index=indices[0],
                                            seed=seed)
        return sample

    def get_codelength(self, indicies):
        if isinstance(self.sampler, RejectionSampler):
            return sum([self.sampler.get_codelength(i) for i in indicies])
        elif isinstance(self.sampler, ImportanceSampler):
            return self.sampler.total_codelength
        else:
            raise NotImplementedError
