import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from rec.models.custom_modules import ReparameterizedConv2D, ReparameterizedConv2DTranspose, GDN

from rec.coding import GaussianCoder, BeamSearchCoder
from rec.coding.samplers import RejectionSampler, ImportanceSampler

from .resnet_vae import ModelError, BidirectionalResidualBlock

tfl = tf.keras.layers
tfk = tf.keras
tfd = tfp.distributions


class LargeResNetVAE(tfk.Model):
    """
    Implements a bidiractional ResNetVAE with more aggressive downsampling and
    2 stochastic residual blocks, which makes it ideal for large images.
    The layout resembles the PLN architecture used by Balle
    """

    AVAILABLE_LIKELIHOODS = [
        "discretized_logistic",
        "gaussian",
        "laplace"
    ]

    def __init__(self,
                 sampler,
                 sampler_args={},
                 coder_args={},
                 use_gdn = True,
                 distribution="gaussian",
                 likelihood_function="laplace",
                 learn_likelihood_scale=False,
                 first_kernel_size=(5, 5),
                 first_strides=(2, 2),
                 kernel_size=(5, 5),
                 strides=(2, 2),
                 first_deterministic_filters=160,
                 second_deterministic_filters=160,
                 first_stochastic_filters=128,
                 second_stochastic_filters=32,
                 kl_per_partition=10,
                 latent_size="variable",
                 ema_decay=0.999,
                 name="resnet_vae",
                 **kwargs):
        super().__init__(name=name,
                         **kwargs)

        # ---------------------------------------------------------------------
        # Assign hyperparamteres
        # ---------------------------------------------------------------------

        self.distribution = distribution

        self.sampler_name = str(sampler)
        self.learn_likelihood_scale = learn_likelihood_scale

        if likelihood_function not in self.AVAILABLE_LIKELIHOODS:
            raise ModelError(f"Likelihood function must be one of: {self.AVAILABLE_LIKELIHOODS}! "
                             f"({likelihood_function} was given).")

        self._likelihood_function = likelihood_function

        self.first_kernel_size = first_kernel_size
        self.first_strides = first_strides

        self.kernel_size = kernel_size
        self.strides = strides
        self.first_stochastic_filters = first_stochastic_filters
        self.first_deterministic_filters = first_deterministic_filters
        self.second_stochastic_filters = second_stochastic_filters
        self.second_deterministic_filters = second_deterministic_filters

        self.kl_per_partition = kl_per_partition
        # Decay for exponential moving average update to variables
        self.ema_decay = tf.cast(ema_decay, tf.float32)

        # ---------------------------------------------------------------------
        # Create parameters
        # ---------------------------------------------------------------------
        self.likelihood_log_scale = tf.Variable(0.,
                                                name="likelihood_log_scale",
                                                trainable=self.learn_likelihood_scale)

        # ---------------------------------------------------------------------
        # Create ResNet Layers
        # ---------------------------------------------------------------------
        # The first deterministic inference block downsamples 8x8
        # Note: we don't apply an ELU at the end of the block, this will happen
        # in the residual block
        self.first_infer_block = [
            ReparameterizedConv2D(kernel_size=self.first_kernel_size,
                                  strides=self.first_strides,
                                  filters=self.first_deterministic_filters,
                                  padding="same"),

            (GDN(inverse=False, name="inf_gdn_0") if use_gdn else tf.nn.elu),

            ReparameterizedConv2D(kernel_size=self.kernel_size,
                                  strides=self.strides,
                                  filters=self.first_deterministic_filters,
                                  padding="same"),

            (GDN(inverse=False, name="inf_gdn_1") if use_gdn else tf.nn.elu),

            ReparameterizedConv2D(kernel_size=self.kernel_size,
                                  strides=self.strides,
                                  filters=self.first_deterministic_filters,
                                  padding="same"),

            (GDN(inverse=False, name="inf_gdn_2") if use_gdn else tf.nn.elu),

            ReparameterizedConv2D(kernel_size=self.kernel_size,
                                  strides=self.strides,
                                  filters=self.first_deterministic_filters,
                                  padding="same"),
        ]

        # The first deterministic generative block is the pseudoinverse of the inference block
        self.first_gen_block = [
            tf.nn.elu,
            ReparameterizedConv2DTranspose(kernel_size=self.kernel_size,
                                           strides=self.strides,
                                           filters=self.first_deterministic_filters,
                                           padding="same"),

            (GDN(inverse=True, name="gen_gdn_0") if use_gdn else tf.nn.elu),

            ReparameterizedConv2DTranspose(kernel_size=self.kernel_size,
                                           strides=self.strides,
                                           filters=self.first_deterministic_filters,
                                           padding="same"),

            (GDN(inverse=True, name="gen_gdn_1") if use_gdn else tf.nn.elu),

            ReparameterizedConv2DTranspose(kernel_size=self.kernel_size,
                                           strides=self.strides,
                                           filters=self.first_deterministic_filters,
                                           padding="same"),

            (GDN(inverse=True, name="gen_gdn_2") if use_gdn else tf.nn.elu),

            ReparameterizedConv2DTranspose(kernel_size=self.first_kernel_size,
                                           strides=self.first_strides,
                                           filters=3,
                                           padding="same"),
        ]

        # The second deterministic inference block downsamples by another 4x4
        self.second_infer_block = [
            ReparameterizedConv2D(kernel_size=(3, 3),
                                  strides=(1, 1),
                                  filters=self.second_deterministic_filters,
                                  padding="same"),
            tf.nn.elu,
            ReparameterizedConv2D(kernel_size=self.kernel_size,
                                  strides=self.strides,
                                  filters=self.second_deterministic_filters,
                                  padding="same"),
            tf.nn.elu,
            ReparameterizedConv2D(kernel_size=self.kernel_size,
                                  strides=self.strides,
                                  filters=self.second_deterministic_filters,
                                  padding="same"),
        ]

        self.second_gen_block = [
            tf.nn.elu,
            ReparameterizedConv2DTranspose(kernel_size=self.kernel_size,
                                           strides=self.strides,
                                           filters=self.second_deterministic_filters,
                                           padding="same"),
            tf.nn.elu,
            ReparameterizedConv2DTranspose(kernel_size=self.kernel_size,
                                           strides=self.strides,
                                           filters=self.second_deterministic_filters,
                                           padding="same"),
            tf.nn.elu,
            ReparameterizedConv2DTranspose(kernel_size=(3, 3),
                                           strides=(1, 1),
                                           filters=self.first_deterministic_filters,
                                           padding="same")
        ]

        # Create Stochastic Residual Blocks
        self.first_residual_block = BidirectionalResidualBlock(
            stochastic_filters=self.first_stochastic_filters,
            deterministic_filters=self.first_deterministic_filters,
            sampler=self.sampler_name,
            sampler_args=sampler_args,
            coder_args=coder_args,
            distribution=distribution,
            kernel_size=self.kernel_size,
            is_last=False,
            use_iaf=False,
            kl_per_partition=self.kl_per_partition,
            name=f"resnet_block_1"
        )

        self.second_residual_block = BidirectionalResidualBlock(
            stochastic_filters=self.second_stochastic_filters,
            deterministic_filters=self.second_deterministic_filters,
            sampler=self.sampler_name,
            sampler_args=sampler_args,
            coder_args=coder_args,
            distribution=distribution,
            kernel_size=self.kernel_size,
            is_last=True,
            use_iaf=False,
            kl_per_partition=self.kl_per_partition,
            name=f"resnet_block_2"
        )

        self.residual_blocks = [self.first_residual_block,
                                self.second_residual_block]

        # Likelihood distribution
        self.likelihood_dist = None

        # Likelihood of the most recent sample
        self.log_likelihood = -np.inf

        # this variable will allow us to perform Empirical Bayes on the first prior
        # Referred to as "h_top" in both the Kingma and Townsend implementations
        self._generative_base = tf.Variable(tf.zeros(self.second_deterministic_filters),
                                            name="generative_base")

        # ---------------------------------------------------------------------
        # EMA shadow variables
        # ---------------------------------------------------------------------
        self._ema_shadow_variables = {}

        self._compressor_initialized = tf.Variable(False, name="compressor_initialized", trainable=False)

    def generative_base(self, batch_size, height, width):

        base = tf.reshape(self._generative_base, [1, 1, 1, self.second_deterministic_filters])

        return tf.tile(base, [batch_size, height // 64, width // 64, 1])

    @property
    def likelihood_function(self):

        likelihood_scale = tf.math.exp(self.likelihood_log_scale)

        def discretized_logistic(reference, reconstruction, binsize=1. / 256.):

            # Discretize the output
            discretized_input = tf.math.floor(reference / binsize) * binsize
            discretized_input = (discretized_input - reconstruction) / likelihood_scale

            log_likelihood = tf.nn.sigmoid(discretized_input + binsize / likelihood_scale)
            log_likelihood = log_likelihood - tf.nn.sigmoid(discretized_input)

            log_likelihood = tf.math.log(log_likelihood + 1e-7)
            return tf.reduce_sum(log_likelihood, [1, 2, 3])

        def gaussian_log_prob(reference, reconstruction):
            likelihood = tfd.Normal(loc=reconstruction, scale=likelihood_scale)
            return tf.reduce_sum(likelihood.log_prob(reference), [1, 2, 3])

        def laplace_log_prob(reference, reconstruction):
            likelihood = tfd.Laplace(loc=reconstruction, scale=likelihood_scale)

            return tf.reduce_sum(likelihood.log_prob(reference), [1, 2, 3])

        if self._likelihood_function == "discretized_logistic":
            return discretized_logistic

        elif self._likelihood_function == "gaussian":
            return gaussian_log_prob

        elif self._likelihood_function == "laplace":
            return laplace_log_prob
        else:
            raise NotImplementedError

    def call(self, tensor, binsize=1 / 256.0):
        input = tensor
        batch_size, height, width, _ = input.shape

        # ---------------------------------------------------------------------
        # Perform Inference Pass
        # ---------------------------------------------------------------------
        for layer in self.first_infer_block:
            tensor = layer(tensor)

        # Pass through the first stochastic residual block
        tensor = self.first_residual_block(tensor, inference_pass=True)

        for layer in self.second_infer_block:
            tensor = layer(tensor)

        # Pass through the second residual block
        self.second_residual_block(tensor, inference_pass=True)
        # ---------------------------------------------------------------------
        # Perform Generative Pass
        # ---------------------------------------------------------------------
        tensor = self.generative_base(batch_size, height, width)

        # Pass through second residual block
        tensor = self.second_residual_block(tensor, inference_pass=False)

        # Second deterministic generative block
        for layer in self.second_gen_block:
            tensor = layer(tensor)

        # Pass through first residual block
        tensor = self.first_residual_block(tensor, inference_pass=False)

        for layer in self.first_gen_block:
            tensor = layer(tensor)

        reconstruction = tf.clip_by_value(tensor, -0.5 + 1. / 512., 0.5 - 1. / 512.)

        # Calculate log likelihood
        log_likelihood = self.likelihood_function(input, reconstruction)
        self.log_likelihood = tf.reduce_mean(log_likelihood)

        # If it's the initialization round, create our EMA shadow variables
        if not self.is_ema_variables_initialized:
            self.create_ema_variables()

        return reconstruction + 0.5

    def kl_divergence(self, empirical=False, minimum_kl=0., reduce=True):

        kls = [res_block.kl_divergence(empirical=empirical, minimum_kl=minimum_kl)
               for res_block in self.residual_blocks]

        if reduce:
            return tf.reduce_sum(kls)
        else:
            return kls

    @property
    def is_ema_variables_initialized(self):
        return len(self._ema_shadow_variables) > 0

    def create_ema_variables(self):
        """
        Creates a shadow copy of every trainable variable. These shadow variables are updated at every training
        iteration using an exponential moving average rule. The EMA variables can then be swapped in for the
        real values at evaluation time, as they supposedly give better performance.
        :return:
        """

        # If the EMA variables have been created already, just skip
        if self.is_ema_variables_initialized:
            return

        self._ema_shadow_variables = {v.name: tf.Variable(v,
                                                          name=f"{v.name}/exponential_moving_average",
                                                          trainable=False)
                                      for v in self.trainable_variables}

    def update_ema_variables(self):
        """
        Update the EMA variables with the latest value of all the current trainable variables.

        This implementation is based on tf.compat.v1.train.ExponentialMovingAverage:
        https://github.com/tensorflow/tensorflow/blob/e5bf8de410005de06a7ff5393fafdf832ef1d4ad/tensorflow/python/training/moving_averages.py#L35
        :return:
        """
        if not self.is_ema_variables_initialized:
            raise ModelError("EMA variables haven't been created yet, since the model has not been initialized yet!")

        for v in self.trainable_variables:
            ema_var = self._ema_shadow_variables[v.name]
            ema_var.assign_sub((1.0 - self.ema_decay) * (ema_var - v))

    def swap_in_ema_variables(self):
        """
        Swap in the EMA shadow variables in place of the real ones for evaluation.
        NOTE: Once the EMA variables have been swapped in, there is no way of swapping back!
        :return:
        """
        if not self.is_ema_variables_initialized:
            raise ModelError("EMA variables haven't been created yet, since the model has not been initialized yet!")

        for v in self.trainable_variables:
            v.assign(self._ema_shadow_variables[v.name])

    # =========================================================================
    # Compression
    # =========================================================================

    def update_coders(self, images):
        # To initialize the coders, we first perform a forward pass with the supplied images.
        # This will set the posteriors and priors in the residual blocks
        self(images)

        for res_block in self.residual_blocks:
            res_block.update_coders()

        self._compressor_initialized.assign(True)

    def compress(self, image, seed, update_sampler=False):

        batch_size, height, width, _ = image.shape
        tensor = image

        block_indices = []

        # ---------------------------------------------------------------------
        # Perform Inference Pass
        # ---------------------------------------------------------------------
        for layer in self.first_infer_block:
            tensor = layer(tensor)

        # Pass through the first stochastic residual block
        tensor = self.first_residual_block(tensor, inference_pass=True)

        for layer in self.second_infer_block:
            tensor = layer(tensor)

        # Pass through the second residual block
        self.second_residual_block(tensor, inference_pass=True)
        # ---------------------------------------------------------------------
        # Perform Generative Pass
        # ---------------------------------------------------------------------
        tensor = self.generative_base(batch_size, height, width)

        # Pass through second residual block
        indices, tensor = self.second_residual_block(tensor,
                                                     inference_pass=False,
                                                     encoder_args={"seed": seed,
                                                                   "update_sampler": update_sampler})

        block_indices.append(indices)

        # Second deterministic generative block
        for layer in self.second_gen_block:
            tensor = layer(tensor)

        # Pass through first residual block
        indices, tensor = self.first_residual_block(tensor,
                                                    inference_pass=False,
                                                    encoder_args={"seed": seed,
                                                                  "update_sampler": update_sampler})

        block_indices.append(indices)

        for layer in self.first_gen_block:
            tensor = layer(tensor)

        reconstruction = tf.clip_by_value(tensor, -0.5 + 1. / 512., 0.5 - 1. / 512.)

        # Discretized Logistic Likelihood
        log_likelihood = self.likelihood_function(image, reconstruction)
        self.log_likelihood = tf.reduce_mean(log_likelihood)

        return block_indices, reconstruction + 0.5

    def get_codelength(self, compressed_codes):
        codelength = 0.
        for resnet_block, compressed_code in zip(self.residual_blocks, compressed_codes):
            codelength += resnet_block.coder.get_codelength(compressed_code)
        return codelength

    def decompress(self, image_shape, block_indices, seed, lossless=False):

        batch_size = 1
        height, width, _ = image_shape

        tensor = self.generative_base(batch_size, height, width)

        # Pass through second residual block
        tensor = self.second_residual_block(tensor,
                                            inference_pass=False,
                                            decoder_args={"seed": seed,
                                                          "indices": block_indices[0]})

        # Second deterministic generative block
        for layer in self.second_gen_block:
            tensor = layer(tensor)

        # Pass through first residual block
        tensor = self.first_residual_block(tensor,
                                           inference_pass=False,
                                           decoder_args={"seed": seed,
                                                         "indices": block_indices[1]})

        for layer in self.first_gen_block:
            tensor = layer(tensor)

        reconstruction = tf.clip_by_value(tensor, -0.5 + 1. / 512., 0.5 - 1. / 512.)

        return reconstruction + 0.5
