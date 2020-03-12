import abc

import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np

from .utils import CodingError
from .samplers import ImportanceSampler, RejectionSampler

tfl = tf.keras.layers
tfd = tfp.distributions


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
    coder_var = coder.scale ** 2
    auxiliary_coder = tfd.Normal(loc=tf.zeros_like(coder.loc), scale=tf.sqrt(auxiliary_var))

    return auxiliary_coder


def get_auxiliary_target(target, coder, auxiliary_var):
    target_var = target.scale ** 2
    coder_var = coder.scale ** 2

    # The auxiliary target is q(A_i | Z) and the auxiliary coder is p(A_i | Z)
    auxiliary_target_mean = (target.loc - coder.loc) * auxiliary_var / coder.scale

    auxiliary_target_variance = target_var * (auxiliary_var ** 2 / coder_var ** 2)
    auxiliary_target_variance += auxiliary_target_variance * (coder_var - auxiliary_var) / coder_var

    auxiliary_target = tfd.Normal(loc=auxiliary_target_mean, scale=tf.sqrt(auxiliary_target_variance))

    return auxiliary_target


def get_conditional_coder(coder, auxiliary_var, auxiliary_sample):
    coder_var = coder.scale ** 2

    new_coder = tfd.Normal(loc=coder.loc + auxiliary_sample,
                           scale=tf.sqrt(coder_var - auxiliary_var))

    return new_coder


def get_conditional_target(target, coder, auxiliary_var, auxiliary_sample):
    target_var = target.scale ** 2
    coder_var = coder.scale ** 2

    # Calculates q(Z | A_i) \propto q(A_i | Z)p(Z)
    # Calculate the mean
    target_mean_numerator = coder.loc + auxiliary_sample * target_var * coder_var
    target_mean_numerator += (target.loc - coder.loc) * (coder_var - auxiliary_var) * coder_var

    target_mean_denominator = target_var * auxiliary_var + coder_var * (coder_var - auxiliary_var)

    new_target_mean = target_mean_numerator / target_mean_denominator

    # Calculate the variance
    target_var_numerator = target_var * coder_var * (coder_var - auxiliary_var)

    target_var_denominator = auxiliary_var * target_var + coder_var * (coder_var - auxiliary_var)

    new_target_var = target_var_numerator / target_var_denominator

    # Instantiate distributions
    new_target = tfd.Normal(loc=new_target_mean,
                            scale=tf.sqrt(new_target_var))

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
        # Create parameters for the auxiliary variables
        # ---------------------------------------------------------------------

        # Calculate the KL between the target and coding distributions
        self.kl = tfd.kl_divergence(self.target_dist, self.coding_dist)
        self.total_kl = tf.reduce_sum(self.kl)

        self.num_aux_variables = 1 + tf.cast(tf.math.floor(self.total_kl / self.kl_per_partition), tf.int32)

        # These will always be forward transformed using a sigmoid to be in (0, 1)
        # The auxiliary variables are always scaled w.r.t the coding distribution, i.e.
        # Var[A_i] = R_i * Var_{Z_i ~ P(Z_i)}[Z_i]
        self.reparameterized_aux_variable_variance_ratios = tf.Variable(-2 * tf.ones(self.num_aux_variables),
                                                                        name="reparameterized_auxiliary_variable_variance_ratios")

        self._initialized = tf.Variable(False,
                                        name="coder_initialized",
                                        trainable=False)

    @property
    def aux_variable_variance_ratios(self):
        return tf.nn.sigmoid(self.reparameterized_aux_variable_variance_ratios)

    def initialize(self, tolerance=1e-4, max_iters=500):
        print("Initializing encoder!")

        target_dist = self.target_dist
        coding_dist = self.coding_dist

        # Perform intialization for each possible KL ratio
        for ratio in range(self.num_aux_variables, 1, -1):
            optimizer = tf.optimizers.SGD(learning_rate=0.001)

            # Optimize the current ratio using SGD
            prev_loss = np.inf

            for _ in range(max_iters):
                with tf.GradientTape() as tape:
                    auxiliary_variance = self.aux_variable_variance_ratios[ratio] * self.coding_dist.scale ** 2
                    aux_target = get_auxiliary_target(target=target_dist,
                                                      coder=coding_dist,
                                                      auxiliary_var=auxiliary_variance)

                    aux_coder = get_auxiliary_coder(coder=coding_dist,
                                                    auxiliary_var=auxiliary_variance)

                    # Get the KL between q(A_i | Z) and p(A_i | Z)
                    auxiliary_kl = tf.reduce_sum(tfd.kl_divergence(aux_target, aux_coder))

                    # Make a quadratic loss
                    kl_loss = (auxiliary_kl - self.total_kl / ratio) ** 2

                gradient = tape.gradient(kl_loss, self.aux_variable_variance_ratios[ratio])
                optimizer.apply_gradient(zip(gradient, [self.aux_variable_variance_ratios[ratio]]))

                # Early stop if the loss decreases less than the tolerance
                if tf.abs(prev_loss - kl_loss) < tolerance:
                    break

                prev_loss = kl_loss

            # Once the optimization is finished, calculate the new target and coding distributions
            auxiliary_sample = aux_target.sample()
            target_dist = get_conditional_target(target=target_dist,
                                                 coder=coding_dist,
                                                 auxiliary_var=auxiliary_variance,
                                                 auxiliary_sample=auxiliary_sample)

            coding_dist = get_conditional_coder(coder=coding_dist,
                                                auxiliary_var=auxiliary_variance,
                                                auxiliary_sample=auxiliary_sample)

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
