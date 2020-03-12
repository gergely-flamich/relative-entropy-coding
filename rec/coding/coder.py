import abc

import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np

from tqdm import trange

from .utils import CodingError
from .samplers import ImportanceSampler, RejectionSampler

from .rejection_sampling import get_aux_distribution, get_conditionals

tfl = tf.keras.layers
tfd = tfp.distributions

square = tf.math.square
log_add_exp = tfp.math.log_add_exp
log_sub_exp = tfp.math.log_sub_exp

log = tf.math.log
exp = tf.exp


class Encoder(tfl.Layer, abc.ABC):

    def __init__(self,
                 target_dist,
                 coding_dist,
                 name="encoder",
                 **kwargs):
        super().__init__(name=name,
                         **kwargs)

        self.target_dist = target_dist
        self.coding_dist = coding_dist


class Decoder(tfl.Layer, abc.ABC):

    def __init__(self,
                 coding_dist,
                 name="decoder",
                 **kwargs):
        super().__init__(name=name,
                         **kwargs)

        self.coding_dist = coding_dist


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
                 target_dist,
                 coding_dist,
                 kl_per_partition=10.0,
                 sampler="importance",
                 name="gaussian_encoder",
                 **kwargs):

        super().__init__(target_dist=target_dist,
                         coding_dist=coding_dist,
                         name=name,
                         **kwargs)

        # ---------------------------------------------------------------------
        # Assign parameters
        # ---------------------------------------------------------------------

        if sampler not in self.AVAILABLE_SAMPLERS:
            raise CodingError(f"Sampler must be one of {self.AVAILABLE_SAMPLERS}, but {sampler} was given!")

        self.sampler = self.AVAILABLE_SAMPLERS[sampler]()

        self.kl_per_partition = tf.cast(kl_per_partition, tf.float32)

        # ---------------------------------------------------------------------
        # Standardize target for numerical stability
        # ---------------------------------------------------------------------
        self.std_target_dist = tfd.Normal(loc=(self.target_dist.loc - self.coding_dist.loc) / self.coding_dist.scale,
                                          scale=self.target_dist.scale / self.coding_dist.scale)
        self.std_coding_dist = tfd.Normal(loc=tf.zeros_like(self.coding_dist.loc),
                                          scale=tf.ones_like(self.coding_dist.scale))

        # ---------------------------------------------------------------------
        # Create parameters for the auxiliary variables
        # ---------------------------------------------------------------------

        # Calculate the KL between the target and coding distributions
        self.kl = tfd.kl_divergence(self.target_dist, self.coding_dist)
        self.total_kl = tf.reduce_sum(self.kl)

        self.num_aux_variables = 1 + tf.cast(tf.math.floor(self.total_kl / self.kl_per_partition), tf.int32)

        # These will always be forward transformed using a sigmoid to be in (0, 1)
        # The auxiliary variables are always scaled w.r.t the coding distribution, i.e.
        # Var[A_i] = R_i * Var_{Z_i ~ P(Z_i)}[Z_i]
        self.reparameterized_aux_variable_variance_ratios = []

        self._initialized = tf.Variable(False,
                                        name="coder_initialized",
                                        trainable=False)

    @property
    def aux_variable_variance_ratios(self):
        return tf.nn.sigmoid(tf.stack(self.reparameterized_aux_variable_variance_ratios, axis=0))

    def initialize(self, tolerance=1e-5, max_iters=500, learning_rate=0.1):
        print(f"Initializing encoder! Total KL to break: {self.total_kl:.4f}")

        target_dist = self.target_dist
        coding_dist = self.coding_dist

        reparameterized_aux_variable_var_ratios = []

        # Perform intialization for each possible KL ratio
        for ratio in range(self.num_aux_variables, 1, -1):

            total_kl = tf.reduce_sum(tfd.kl_divergence(target_dist, coding_dist))

            init = -2.
            if len(reparameterized_aux_variable_var_ratios) > 0:
                # This heuristic usually works very well, and the optimizer converges in
                # approximately 100 iterations
                init = reparameterized_aux_variable_var_ratios[-1] - 0.5

            reparameterized_aux_variable_var_ratio = tf.Variable(init,
                                                                 name=f"reparameterized_aux_var_ratio/{ratio}")

            optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

            # Optimize the current ratio using SGD
            prev_loss = np.inf

            with trange(max_iters) as progress_bar:
                for _ in progress_bar:
                    with tf.GradientTape() as tape:
                        aux_variable_variance_ratio = tf.nn.sigmoid(reparameterized_aux_variable_var_ratio)

                        auxiliary_variance = aux_variable_variance_ratio * self.coding_dist.scale ** 2
                        aux_target = get_auxiliary_target(target=target_dist,
                                                          coder=coding_dist,
                                                          auxiliary_var=auxiliary_variance)

                        aux_coder = get_auxiliary_coder(coder=coding_dist,
                                                        auxiliary_var=auxiliary_variance)

                        # Get the KL between q(A_i | Z) and p(A_i | Z)
                        auxiliary_kl = tf.reduce_sum(tfd.kl_divergence(aux_target, aux_coder))

                        # Make a quadratic loss
                        kl_loss = square(auxiliary_kl - total_kl / tf.cast(ratio, tf.float32))

                    gradient = tape.gradient(kl_loss, reparameterized_aux_variable_var_ratio)
                    optimizer.apply_gradients([(gradient, reparameterized_aux_variable_var_ratio)])

                    # Early stop if the loss decreases less than the tolerance
                    if tf.abs(prev_loss - kl_loss) < tolerance:
                        break

                    prev_loss = kl_loss

                    progress_bar.set_description(f"Ratio {ratio}: {aux_variable_variance_ratio:.8f}, "
                                                 f"KL: {total_kl:.3f}, Aux KL: {auxiliary_kl:.3f}, "
                                                 f"Target KL: {(total_kl / tf.cast(ratio, tf.float32)):.3f}, "
                                                 f"KL loss: {kl_loss:.3f}")

            # Once the optimization is finished, calculate the new target and coding distributions
            auxiliary_sample = aux_target.sample()
            target_dist = get_conditional_target(target=target_dist,
                                                 coder=coding_dist,
                                                 auxiliary_var=auxiliary_variance,
                                                 auxiliary_sample=auxiliary_sample)

            coding_dist = get_conditional_coder(coder=coding_dist,
                                                auxiliary_var=auxiliary_variance,
                                                auxiliary_sample=auxiliary_sample)

            reparameterized_aux_variable_var_ratios.append(reparameterized_aux_variable_var_ratio.value())

        self._initialized.assign(True)

    def call(self, seed):

        if not self._initialized:
            raise CodingError("Coder has not been initialized yet, please call initialize() first!")

        target_dist = self.target_dist
        coding_dist = self.coding_dist

        indices = []
        aggregate_sample = 0.

        for aux_variable_variance_ratio in self.aux_variable_variance_ratios:
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

        return indices, aggregate_sample


class GaussianDecoder(Decoder):
    AVAILABLE_SAMPLERS = {
        "importance": ImportanceSampler,
        "rejection": RejectionSampler
    }

    def __init__(self,
                 coding_dist,
                 auxiliary_variable_variance_ratios,
                 sampler="importance",
                 name="gaussian_decoder",
                 **kwargs):
        super().__init__(name=name,
                         coding_dist=coding_dist,
                         **kwargs)

        self.auxiliary_variable_variance_ratios = auxiliary_variable_variance_ratios

        if sampler not in self.AVAILABLE_SAMPLERS:
            raise CodingError(f"Sampler must be one of {self.AVAILABLE_SAMPLERS}, but {sampler} was given!")

        self.sampler = self.AVAILABLE_SAMPLERS[sampler]()

    def call(self, indices, seed):

        aggregate_sample = 0.

        coding_dist = self.coding_dist

        for aux_variable_variance_ratio, index in zip(self.aux_variable_variance_ratios, indices):
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

        return sample
