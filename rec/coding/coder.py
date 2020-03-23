import abc

import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np

from tqdm import trange

from rec.coding.utils import CodingError
from rec.coding.samplers import ImportanceSampler, RejectionSampler

from rec.coding.rejection_sampling import get_aux_distribution, get_conditionals

tfl = tf.keras.layers
tfd = tfp.distributions

square = tf.math.square
# TODO these functions don't exist, fix please
log_add_exp = None  # tfp.math.log_add_exp
log_sub_exp = None  # tfp.math.log_sub_exp

log = tf.math.log
exp = tf.exp


def sigmoid_inverse(x):
    if tf.reduce_any(x < 0.) or tf.reduce_any(x > 1.):
        raise ValueError(f"x = {x} was not in the sigmoid function's range ([0, 1])!")
    x = tf.clip_by_value(x, 1e-10, 1 - 1e-10)

    return tf.math.log(x) - tf.math.log(1. - x)


class Encoder(tfl.Layer, abc.ABC):

    def __init__(self,
                 name="encoder",
                 **kwargs):
        super().__init__(name=name,
                         **kwargs)


class Decoder(tfl.Layer, abc.ABC):

    def __init__(self,
                 name="decoder",
                 **kwargs):
        super().__init__(name=name,
                         **kwargs)


def get_auxiliary_coder(coder, auxiliary_var):
    auxiliary_coder = tfd.Normal(loc=tf.zeros_like(coder.loc), scale=tf.sqrt(auxiliary_var))

    return auxiliary_coder


# def get_auxiliary_target(target, coder, auxiliary_var):
#     target_var = square(target.scale)
#     coder_var = square(coder.scale)
#
#     aux_var_over_coder_var = auxiliary_var / coder_var
#
#     # The auxiliary target is q(A_i | Z) and the auxiliary coder is p(A_i | Z)
#     auxiliary_target_mean = (target.loc - coder.loc) * aux_var_over_coder_var
#
#     auxiliary_target_variance = target_var * square(aux_var_over_coder_var)
#     auxiliary_target_variance += (coder_var - auxiliary_var) * aux_var_over_coder_var
#
#     auxiliary_target = tfd.Normal(loc=auxiliary_target_mean, scale=tf.sqrt(auxiliary_target_variance))
#
#     return auxiliary_target
def get_auxiliary_target(target, coder, auxiliary_var, eps=1e-7):
    log_target_var = 2 * log(target.scale + eps)
    log_coder_var = 2 * log(coder.scale + eps)
    log_aux_var = log(auxiliary_var + eps)

    log_aux_var_over_coder_var = log_aux_var - log_coder_var

    # The auxiliary target is q(A_i | Z) and the auxiliary coder is p(A_i | Z)
    auxiliary_target_mean = (target.loc - coder.loc) * exp(log_aux_var_over_coder_var)

    log_auxiliary_target_variance = log_add_exp(log_target_var + 2 * log_aux_var_over_coder_var,
                                                log_sub_exp(log_coder_var, log_aux_var) + log_aux_var_over_coder_var)

    auxiliary_target = tfd.Normal(loc=auxiliary_target_mean, scale=exp(0.5 * log_auxiliary_target_variance))

    return auxiliary_target


def get_conditional_coder(coder, auxiliary_var, auxiliary_sample):
    coder_var = square(coder.scale)

    new_coder = tfd.Normal(loc=coder.loc + auxiliary_sample,
                           scale=tf.sqrt(coder_var - auxiliary_var))

    return new_coder


# def get_conditional_target(target, coder, auxiliary_var, auxiliary_sample):
#     target_var = square(target.scale)
#     coder_var = square(coder.scale)
#
#     # Calculates q(Z | A_i) \propto q(A_i | Z)p(Z)
#
#     target_denominator = target_var * auxiliary_var + coder_var * (coder_var - auxiliary_var)
#
#     # Calculate the mean
#     target_mean_numerator = auxiliary_sample * target_var * coder_var
#     target_mean_numerator += (target.loc - coder.loc) * (coder_var - auxiliary_var) * coder_var
#
#     new_target_mean = coder.loc + target_mean_numerator / target_denominator
#
#     # Calculate the variance
#     target_var_numerator = target_var * coder_var * (coder_var - auxiliary_var)
#
#     new_target_var = target_var_numerator / target_denominator
#
#     # Instantiate distributions
#     new_target = tfd.Normal(loc=new_target_mean,
#                             scale=tf.sqrt(new_target_var))
#
#     return new_target
def get_conditional_target(target, coder, auxiliary_var, auxiliary_sample, eps=1e-7):
    log_target_var = 2 * log(target.scale + eps)
    log_coder_var = 2 * log(coder.scale + eps)
    log_aux_var = log(auxiliary_var + eps)

    # Calculates q(Z | A_i) \propto q(A_i | Z)p(Z)
    log_target_denominator = log_add_exp(log_target_var + log_aux_var,
                                         log_coder_var + log_sub_exp(log_coder_var, log_aux_var))

    # Calculate the mean
    target_mean_term_1 = auxiliary_sample * exp(log_target_var + log_coder_var - log_target_denominator)

    log_target_mean_term_2 = log_sub_exp(log_coder_var, log_aux_var) + log_coder_var - log_target_denominator
    target_mean_term_2 = (target.loc - coder.loc) * exp(log_target_mean_term_2)

    new_target_mean = coder.loc + target_mean_term_1 + target_mean_term_2

    # Calculate the variance
    log_target_var_numerator = log_target_var + log_coder_var + log_sub_exp(log_coder_var,
                                                                            log_aux_var)

    log_new_target_var = log_target_var_numerator - log_target_denominator

    # Instantiate distributions
    new_target = tfd.Normal(loc=new_target_mean,
                            scale=exp(0.5 * log_new_target_var))

    return new_target


class GaussianEncoder(Encoder):
    AVAILABLE_SAMPLERS = {
        "importance": ImportanceSampler,
        "rejection": RejectionSampler
    }

    def __init__(self,
                 kl_per_partition=10.0,
                 sampler="importance",
                 name="gaussian_encoder",
                 **kwargs):

        super().__init__(name=name,
                         **kwargs)

        # ---------------------------------------------------------------------
        # Assign parameters
        # ---------------------------------------------------------------------

        if sampler not in self.AVAILABLE_SAMPLERS:
            raise CodingError(f"Sampler must be one of {self.AVAILABLE_SAMPLERS}, but {sampler} was given!")

        self.sampler = self.AVAILABLE_SAMPLERS[sampler]()

        self.kl_per_partition = tf.cast(kl_per_partition, tf.float32)

        # ---------------------------------------------------------------------
        # Create parameters for the auxiliary variables
        # ---------------------------------------------------------------------

        # The auxiliary variables are always scaled w.r.t the coding distribution, i.e.
        # Var[A_i] = R_i * Var_{Z_i ~ P(Z_i)}[Z_i]
        self.sum_aux_variable_variance_ratios = tf.Variable(tf.zeros([0], dtype=tf.float32),
                                                            shape=(None,),
                                                            name="sum_averaged_variance_ratios",
                                                            trainable=False)

        # Counts over how many batch elements we averaged over
        self.average_counts = tf.Variable(tf.zeros([0], dtype=tf.float32),
                                          shape=(None,),
                                          name="average_counts",
                                          trainable=False)

        self._initialized = tf.Variable(False,
                                        name="coder_initialized",
                                        trainable=False)

    @property
    def aux_variable_variance_ratios(self):
        if self.average_counts is None:
            raise CodingError("Coder must be initialized!")

        # The first item is always undefined, since it is implictly the same as the second item
        return self.sum_aux_variable_variance_ratios / self.average_counts

    def update_auxiliary_variance_ratios(self,
                                         target_dist,
                                         coding_dist,
                                         relative_tolerance=1e-5,
                                         absolute_tolerance=1e-4,
                                         max_iters=10000,
                                         learning_rate=0.03):
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

        sum_variance_ratios = np.zeros(max_num_variables, dtype=np.float32)
        average_counts = np.zeros(max_num_variables, dtype=np.float32)

        # Perform intialization for each possible KL ratio
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
            init = 1. / ratio * np.ones(num_elements, dtype=np.float32)

            if ratio < max_num_variables:
                init = tf.maximum(tf.cast(sum_variance_ratios[ratio] / average_counts[ratio] - 0.1, tf.float32),
                                  init)

            init = sigmoid_inverse(init)
            reparameterized_aux_variable_var_ratio = tf.Variable(init)

            # Compensate in the learning rate for the increased loss
            optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

            # Optimize the current ratio using SGD
            prev_loss = np.inf

            with trange(max_iters) as progress_bar:
                for _ in progress_bar:
                    with tf.GradientTape() as tape:
                        aux_variable_variance_ratio = tf.nn.sigmoid(reparameterized_aux_variable_var_ratio)

                        # We reshape, so that broadcasting works nice
                        aux_variable_variance_ratio = tf.reshape(aux_variable_variance_ratio,
                                                                 [-1] + [1] * (tf.rank(target_loc).numpy() - 1))

                        auxiliary_variance = aux_variable_variance_ratio * coder.scale ** 2
                        aux_target = get_auxiliary_target(target=target,
                                                          coder=coder,
                                                          auxiliary_var=auxiliary_variance)

                        aux_coder = get_auxiliary_coder(coder=coder,
                                                        auxiliary_var=auxiliary_variance)

                        # Get the KL between q(A_i | Z) and p(A_i | Z)
                        auxiliary_kl = tf.reduce_sum(tfd.kl_divergence(aux_target, aux_coder),
                                                     axis=data_dims)

                        # Make a quadratic loss
                        kl_loss = square(auxiliary_kl - total_kl / tf.cast(ratio, tf.float32))
                        total_kl_loss = tf.reduce_mean(kl_loss)

                    gradient = tape.gradient(total_kl_loss, reparameterized_aux_variable_var_ratio)
                    optimizer.apply_gradients([(gradient, reparameterized_aux_variable_var_ratio)])

                    # Early stop if the loss decreases less than the tolerance
                    if total_kl_loss < absolute_tolerance and \
                            tf.abs(prev_loss - total_kl_loss) < relative_tolerance:
                        break

                    prev_loss = total_kl_loss

                    ratio_mean, ratio_var = tf.nn.moments(tf.reshape(aux_variable_variance_ratio, [-1]), axes=[0])

                    progress_bar.set_description(f"Ratio {ratio}, {num_elements}/{target_loc.shape[0]} items - "
                                                 f"avg ratio: {ratio_mean:.4f}+-{tf.sqrt(ratio_var):.4f}, "
                                                 f"aux_kl_1: {auxiliary_kl[0]:.3f}, "
                                                 f"target_kl_1: {(total_kl[0] / ratio):.3f}, "
                                                 f"kl_1: {(total_kl[0]):.3f}, "
                                                 f"loss: {total_kl_loss:.3f}")

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

            # Update the statistics we want to persist
            sum_variance_ratios[ratio - 1] = tf.reduce_sum(aux_variable_variance_ratio).numpy()
            average_counts[ratio - 1] = num_elements

        # ---------------------------------------------------------------------
        # Once we have the ratios, we concatenate them, and bind them to the encoder
        # by creating variables
        # ---------------------------------------------------------------------

        # If the variables exist already, we combine them
        num_stored_vars = self.average_counts.value().shape[0]

        # If there are are more dimensions than we previously had, we need to extend our variables
        if num_stored_vars < max_num_variables:
            # Update the indices that exist
            sum_variance_ratios[:num_stored_vars] += self.aux_variable_variance_ratios.numpy()
            average_counts[:num_stored_vars] += self.average_counts.numpy()

        # If the number of dimensions is fewer than we previously had, we do a simple update
        else:
            # Update the indices that exist
            sum_variance_ratios += self.aux_variable_variance_ratios[:max_num_variables].numpy()
            average_counts += self.average_counts[:max_num_variables].numpy()

        # Assign values
        self.average_counts.assign(average_counts)
        self.sum_aux_variable_variance_ratios.assign(sum_variance_ratios)

        self._initialized.assign(True)

    def call(self, target_dist, coding_dist, seed):

        if not self._initialized:
            raise CodingError("Coder has not been initialized yet, please call initialize() first!")

        indices = []
        aggregate_sample = 0.

        total_kl = tf.reduce_sum(tfd.kl_divergence(target_dist, coding_dist))
        num_aux_variables = 1 + tf.cast(tf.math.floor(total_kl / self.kl_per_partition), tf.int32)

        # If there are more auxiliary variables needed than what we are already storing, we update our estimates
        if num_aux_variables > self.average_counts.shape[0]:
            print("More auxiliary variables required, updating internal statistics!")
            self.update_auxiliary_variance_ratios(target_dist=target_dist,
                                                  coding_dist=coding_dist)

        # We iterate from the second entry in the ratios, because the first entry is *implicitly*
        # the same as the second, but in reality it is NaN!
        for aux_variable_variance_ratio in self.aux_variable_variance_ratios[1:num_aux_variables][::-1]:
            auxiliary_var = aux_variable_variance_ratio * coding_dist.scale ** 2

            auxiliary_target = get_auxiliary_target(target=target_dist,
                                                    coder=coding_dist,
                                                    auxiliary_var=auxiliary_var)

            auxiliary_coder = get_auxiliary_coder(coder=coding_dist,
                                                  auxiliary_var=auxiliary_var)

            index, sample = self.sampler.coded_sample(target=auxiliary_target,
                                                      coder=auxiliary_coder,
                                                      seed=seed)

            indices.append(index)

            aggregate_sample = aggregate_sample + sample

            target_dist = get_conditional_target(target=target_dist,
                                                 coder=coding_dist,
                                                 auxiliary_var=auxiliary_var,
                                                 auxiliary_sample=sample)

            coding_dist = get_conditional_coder(coder=coding_dist,
                                                auxiliary_var=auxiliary_var,
                                                auxiliary_sample=sample)

        # Sample the last auxiliary variable
        index, sample = self.sampler.coded_sample(target=target_dist,
                                                  coder=coding_dist,
                                                  seed=seed)

        indices.append(index)

        aggregate_sample = aggregate_sample + sample

        return indices, aggregate_sample


class GaussianDecoder(Decoder):
    AVAILABLE_SAMPLERS = {
        "importance": ImportanceSampler,
        "rejection": RejectionSampler
    }

    def __init__(self,
                 sampler="importance",
                 name="gaussian_decoder",
                 **kwargs):
        super().__init__(name=name,
                         **kwargs)

        if sampler not in self.AVAILABLE_SAMPLERS:
            raise CodingError(f"Sampler must be one of {self.AVAILABLE_SAMPLERS}, but {sampler} was given!")

        self.sampler = self.AVAILABLE_SAMPLERS[sampler]()

    def call(self, indices, seed, aux_variable_variance_ratios):

        aggregate_sample = 0.

        coding_dist = self.coding_dist

        for aux_variable_variance_ratio, index in zip(aux_variable_variance_ratios, indices):
            auxiliary_var = aux_variable_variance_ratio * coding_dist.scale ** 2

            auxiliary_coder = get_auxiliary_coder(coder=coding_dist,
                                                  auxiliary_var=auxiliary_var)

            sample = self.sampler.decode_sample(coder=auxiliary_coder,
                                                sample_index=index,
                                                seed=seed)

            aggregate_sample = aggregate_sample + sample

            coding_dist = get_conditional_coder(coder=coding_dist,
                                                auxiliary_var=auxiliary_var,
                                                auxiliary_sample=sample)

        return aggregate_sample
