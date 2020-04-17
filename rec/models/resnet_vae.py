from typing import Tuple

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from rec.models.custom_modules import ReparameterizedConv2D, ReparameterizedConv2DTranspose, AutoRegressiveMultiConv2D
from rec.coding import GaussianCoder, BeamSearchCoder
from rec.coding.samplers import RejectionSampler, ImportanceSampler

tfl = tf.keras.layers
tfk = tf.keras
tfd = tfp.distributions


class ModelError(Exception):
    pass


class BidirectionalResidualBlock(tfl.Layer):
    """
    Implements a bidirectional Resnet Block
    """

    def __init__(self,
                 stochastic_filters: int,
                 deterministic_filters: int,
                 sampler: str,
                 sampler_args: dict = {},
                 coder_args: dict = {},
                 kernel_size: Tuple[int, int] =(3, 3),
                 use_iaf: bool = False,
                 is_last: bool = False,
                 kl_per_partition = 8.,
                 name: str = "bidirectional_resnet_block",
                 **kwargs):
        super().__init__(name=name,
                         **kwargs)

        # Number of filters for the stochastic layers
        self.stochastic_filters = stochastic_filters

        # Number of filters for the deterministic residual features
        self.deterministic_filters = deterministic_filters
        self.kernel_size = kernel_size

        # If the resnet block is the last one in the VAE, we won't use the final bit of the residual block.
        self.is_last = is_last

        # Use inverse autoregressive flows as the posterior?
        self.use_iaf = use_iaf

        # ---------------------------------------------------------------------
        # Declare layers
        # ---------------------------------------------------------------------
        # Infernce block parts
        self.infer_conv1 = None
        self.infer_conv2 = None

        self.infer_posterior_loc_head = None
        self.infer_posterior_log_scale_head = None

        # Generative block parts
        self.gen_conv1 = None
        self.gen_conv2 = None

        self.prior_loc_head = None
        self.prior_log_scale_head = None

        self.gen_posterior_loc_head = None
        self.gen_posterior_log_scale_head = None

        # ---------------------------------------------------------------------
        # Declare Bidirectional inference components
        # ---------------------------------------------------------------------
        self.infer_posterior_loc = 0.
        self.infer_posterior_log_scale = 0.

        self.gen_posterior_loc = 0.
        self.gen_posterior_log_scale = 0.

        self.prior_loc = 0.
        self.prior_scale = 1.

        # ---------------------------------------------------------------------
        # Distributions associated with the current residual block
        # ---------------------------------------------------------------------
        self.posterior = None
        self.prior = None

        self.infer_iaf_autoregressive_context_conv = None
        self.gen_iaf_autoregressive_context_conv = None
        self.iaf_posterior_multiconv = None

        self.infer_iaf_autoregressive_context = None
        self.gen_iaf_autoregressive_context = None

        self.empirical_kld = 0.

        # ---------------------------------------------------------------------
        # Stuff for compression
        # ---------------------------------------------------------------------
        if sampler == "rejection":
            self.coder = GaussianCoder(sampler=RejectionSampler(**sampler_args),
                                       kl_per_partition=kl_per_partition,
                                       name=f"encoder_for_{self.name}",
                                       **coder_args)
        elif sampler == "importance":
            # Setting alpha=inf will select the sample with
            # the best importance weights
            self.coder = GaussianCoder(sampler=ImportanceSampler(**sampler_args),
                                       kl_per_partition=kl_per_partition,
                                       name=f"encoder_for_{self.name}",
                                       **coder_args)
        elif sampler == "beam_search":
            self.coder = BeamSearchCoder(kl_per_partition=kl_per_partition,
                                         n_beams=sampler_args['n_beams'],
                                         extra_samples=sampler_args['extra_samples'],
                                         name=f"encoder_for_{self.name}",
                                         **coder_args)
        else:
            raise ModelError("Sampler must be one of ['rejection', 'importance', 'beam_search'],"
                             f"but got {sampler}!")

        # ---------------------------------------------------------------------
        # Initialization flag
        # ---------------------------------------------------------------------
        self._initialized = tf.Variable(False, name="resnet_block_initialized", trainable=False)

    @property
    def posterior_loc(self):
        return self.infer_posterior_loc + self.gen_posterior_loc

    @property
    def posterior_scale(self):
        return tf.exp(self.infer_posterior_log_scale + self.gen_posterior_log_scale)

    @property
    def iaf_autoregressive_context(self):
        if not self.use_iaf:
            raise ModelError("IAF contexts only exist when model is in IAF mode!")

        return self.infer_iaf_autoregressive_context + self.gen_iaf_autoregressive_context

    def kl_divergence(self, empirical=False, minimum_kl=0.):

        if self.use_iaf and not empirical:
            raise ModelError("KL divergence cannot be computed analytically when"
                             "using IAFs as posterior!")

        if empirical:
            kld = self.empirical_kld
        else:
            kld = tfd.kl_divergence(self.posterior, self.prior)

        # The parameters are shared per channel, so we first calculate the average
        # across the batch, width and height axes, then apply the minimum KL constraint,
        # and finally sum across the filters
        kld = tf.reduce_mean(tf.reduce_sum(kld, axis=[1, 2]), axis=[0])

        kld = tf.maximum(kld, minimum_kl)

        kld = tf.reduce_sum(kld)

        return kld

    def build(self, input_shape):
        # ---------------------------------------------------------------------
        # Stuff for the inference side
        # ---------------------------------------------------------------------

        if not self.is_last:
            self.infer_conv1 = ReparameterizedConv2D(filters=self.deterministic_filters,
                                                     kernel_size=self.kernel_size,
                                                     strides=(1, 1),
                                                     padding="same")

            self.infer_conv2 = ReparameterizedConv2D(filters=self.deterministic_filters,
                                                     kernel_size=self.kernel_size,
                                                     strides=(1, 1),
                                                     padding="same")

        self.infer_posterior_loc_head = ReparameterizedConv2D(filters=self.stochastic_filters,
                                                              kernel_size=self.kernel_size,
                                                              strides=(1, 1),
                                                              padding="same")

        self.infer_posterior_log_scale_head = ReparameterizedConv2D(filters=self.stochastic_filters,
                                                                    kernel_size=self.kernel_size,
                                                                    strides=(1, 1),
                                                                    padding="same")

        # ---------------------------------------------------------------------
        # Stuff for the generative side
        # Note: In the general case, these should technically be deconvolutions, but
        # in the original implementation the dimensions within a single block do not
        # decrease, hence there is not much point in using the more expensive operation
        # ---------------------------------------------------------------------
        self.gen_conv1 = ReparameterizedConv2D(filters=self.deterministic_filters,
                                               kernel_size=self.kernel_size,
                                               strides=(1, 1),
                                               padding="same")

        self.gen_conv2 = ReparameterizedConv2D(filters=self.deterministic_filters,
                                               kernel_size=self.kernel_size,
                                               strides=(1, 1),
                                               padding="same")

        self.prior_loc_head = ReparameterizedConv2D(filters=self.stochastic_filters,
                                                    kernel_size=self.kernel_size,
                                                    strides=(1, 1),
                                                    padding="same")

        self.prior_log_scale_head = ReparameterizedConv2D(filters=self.stochastic_filters,
                                                          kernel_size=self.kernel_size,
                                                          strides=(1, 1),
                                                          padding="same")

        self.gen_posterior_loc_head = ReparameterizedConv2D(filters=self.stochastic_filters,
                                                            kernel_size=self.kernel_size,
                                                            strides=(1, 1),
                                                            padding="same")

        self.gen_posterior_log_scale_head = ReparameterizedConv2D(filters=self.stochastic_filters,
                                                                  kernel_size=self.kernel_size,
                                                                  strides=(1, 1),
                                                                  padding="same")

        # ---------------------------------------------------------------------
        # If we use IAF posteriors, we need some additional layers
        # ---------------------------------------------------------------------
        if self.use_iaf:
            self.infer_iaf_autoregressive_context_conv = ReparameterizedConv2D(
                filters=self.deterministic_filters,
                kernel_size=self.kernel_size,
                strides=(1, 1),
                padding="same"
            )

            self.gen_iaf_autoregressive_context_conv = ReparameterizedConv2D(
                filters=self.deterministic_filters,
                kernel_size=self.kernel_size,
                strides=(1, 1),
                padding="same"
            )

            self.iaf_posterior_multiconv = AutoRegressiveMultiConv2D(
                convolution_filters=[self.deterministic_filters,
                                     self.deterministic_filters],
                head_filters=[self.stochastic_filters,
                              self.stochastic_filters]
            )

        super().build(input_shape=input_shape)

    def call(self, tensor, inference_pass=True, encoder_args=None, decoder_args=None, eps=1e-7):
        """

        :param tensor: data to be passed through the residual block
        :param inference_pass:
        :return:
        """

        input = tensor

        # First layer in block
        tensor = tf.nn.elu(tensor)
        # ---------------------------------------------------------------------
        # Inference pass
        # ---------------------------------------------------------------------
        if inference_pass:

            # Calculate first part of posterior statistics
            self.infer_posterior_loc = self.infer_posterior_loc_head(tensor)
            self.infer_posterior_log_scale = self.infer_posterior_log_scale_head(tensor)

            if self.use_iaf:
                self.infer_iaf_autoregressive_context = self.infer_iaf_autoregressive_context_conv(tensor)

            # Calculate next set of deterministic features
            if not self.is_last:
                tensor = self.infer_conv1(tensor)
                tensor = tf.nn.elu(tensor)

                tensor = self.infer_conv2(tensor)

        # ---------------------------------------------------------------------
        # Generative pass
        # ---------------------------------------------------------------------
        else:

            # Calculate prior parameters
            self.prior_loc = self.prior_loc_head(tensor)
            self.prior_scale = tf.exp(self.prior_log_scale_head(tensor))

            self.prior = tfd.Normal(loc=self.prior_loc,
                                    scale=self.prior_scale)
            # -----------------------------------------------------------------
            # Training
            # -----------------------------------------------------------------

            # If no latent code is provided, we need to create it
            if encoder_args is None and decoder_args is None:
                # Calculate second part of posterior statistics
                self.gen_posterior_loc = self.gen_posterior_loc_head(tensor)
                self.gen_posterior_log_scale = self.gen_posterior_log_scale_head(tensor)

                # Sample from posterior. The loc and scale are automagically calculated using property methods
                self.posterior = tfd.Normal(loc=self.posterior_loc,
                                            scale=self.posterior_scale)

                if self._initialized:
                    latent_code = self.posterior.sample()
                else:
                    latent_code = self.prior.sample()
                    self._initialized.assign(True)

                post_log_prob = self.posterior.log_prob(latent_code)

                if self.use_iaf:
                    self.gen_iaf_autoregressive_context = self.gen_iaf_autoregressive_context_conv(tensor)

                    context = self.iaf_autoregressive_context

                    iaf_mean, iaf_log_scale = self.iaf_posterior_multiconv(latent_code,
                                                                           context=context)

                    iaf_mean = 0.1 * iaf_mean
                    iaf_log_scale = 0.1 * iaf_log_scale

                    # Update latent code
                    latent_code = (latent_code - iaf_mean) / tf.exp(iaf_log_scale)

                    # Update posterior log probability with IAF's Jacobian logdet
                    post_log_prob = post_log_prob + iaf_log_scale

                # Note: prior log probability needs to be calculated once we passed the latent
                # code through the IAF, since we care about the transformed sample!
                prior_log_prob = self.prior.log_prob(latent_code)

                self.empirical_kld = post_log_prob - prior_log_prob

            # -----------------------------------------------------------------
            # Compression
            # -----------------------------------------------------------------
            if encoder_args is not None:

                # Calculate second part of posterior statistics
                self.gen_posterior_loc = self.gen_posterior_loc_head(tensor)
                self.gen_posterior_log_scale = self.gen_posterior_log_scale_head(tensor)

                # The loc and scale are automagically calculated using property methods
                self.posterior = tfd.Normal(loc=self.posterior_loc,
                                            scale=self.posterior_scale)
                indices, latent_code = self.coder.encode(self.posterior, self.prior, **encoder_args)

            # -----------------------------------------------------------------
            # Decompression
            # -----------------------------------------------------------------
            if decoder_args is not None:
                latent_code = self.coder.decode(self.prior, **decoder_args)

            # Calculate next set of deterministic features for residual block
            tensor = self.gen_conv1(tensor)

            # Concatenate code and generative features. The channels are always the last axis
            tensor = tf.concat([tensor, latent_code], axis=-1)

            tensor = tf.nn.elu(tensor)

            tensor = self.gen_conv2(tensor)

        # Add residual connection. Scaling factor taken from
        # https://github.com/hilloc-submission/hilloc/blob/b89e9c983e3764798e7c6f81f5cfc1d11b349d96/experiments/rvae/model/__init__.py#L116
        tensor = input + 0.1 * tensor

        if encoder_args is not None:
            return indices, tensor

        return tensor

    def update_coders(self):
        self.coder.update_auxiliary_variance_ratios(target_dist=self.posterior,
                                                    coding_dist=self.prior)

    def posterior_log_prob(self, tensor):
        if self.use_iaf:
            raise NotImplementedError

        else:
            return tf.reduce_sum(self.posterior.log_prob(tensor))

    def prior_log_prob(self, tensor):
        return tf.reduce_sum(self.prior.log_prob(tensor))


class BidirectionalResNetVAE(tfk.Model):
    """
    Implements the bidirectional ResNetVAE as described in:
    D. P. Kingma, T. Salimans, R. Jozefowicz, X. Chen, I. Sutskever, and M. Welling.
    Improved variational inference with inverse autoregressive flow.
    In Advances in Neural Information ProcessingSystems (NIPS), 2016.
    """

    AVAILABLE_LIKELIHOODS = [
        "discretized_logistic",
        "gaussian",
        "laplace",
        "ms-ssim"
    ]

    def __init__(self,
                 num_res_blocks,
                 sampler,
                 sampler_args={},
                 coder_args={},
                 likelihood_function="discretized_logistic",
                 learn_likelihood_scale=True,
                 first_kernel_size=(5, 5),
                 first_strides=(2, 2),
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 deterministic_filters=160,
                 stochastic_filters=32,
                 use_iaf=False,
                 kl_per_partition=None,
                 latent_size="variable",
                 ema_decay=0.999,
                 name="resnet_vae",
                 **kwargs):
        super().__init__(name=name,
                         **kwargs)

        # ---------------------------------------------------------------------
        # Assign hyperparamteres
        # ---------------------------------------------------------------------
        self.sampler_name = str(sampler)

        self.num_res_blocks = num_res_blocks

        self.learn_likelihood_scale = learn_likelihood_scale

        if likelihood_function not in self.AVAILABLE_LIKELIHOODS:
            raise ModelError(f"Likelihood function must be one of: {self.AVAILABLE_LIKELIHOODS}! "
                             f"({likelihood_function} was given).")

        self._likelihood_function = likelihood_function

        self.first_kernel_size = first_kernel_size
        self.first_strides = first_strides

        self.kernel_size = kernel_size
        self.strides = strides
        self.stochastic_filters = stochastic_filters
        self.deterministic_filters = deterministic_filters

        self.use_iaf = use_iaf

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
        self.first_infer_conv = ReparameterizedConv2D(kernel_size=self.first_kernel_size,
                                                      strides=self.first_strides,
                                                      filters=self.deterministic_filters,
                                                      padding="same")

        self.last_gen_conv = ReparameterizedConv2DTranspose(kernel_size=self.first_kernel_size,
                                                            strides=self.first_strides,
                                                            filters=3,
                                                            padding="same")

        # We create these in topological order.
        # This means that residual_blocks[0] will have the bottom-most stochastic layer
        # And residual_blocks[-1] will have the top-most one, the output of which should be passed to last_gen_conv
        self.residual_blocks = [BidirectionalResidualBlock(stochastic_filters=self.stochastic_filters,
                                                           deterministic_filters=self.deterministic_filters,
                                                           sampler=self.sampler_name,
                                                           sampler_args=sampler_args,
                                                           coder_args=coder_args,
                                                           kernel_size=self.kernel_size,
                                                           is_last=res_block_idx == 0,  # Declare last residual block
                                                           use_iaf=self.use_iaf,
                                                           kl_per_partition=self.kl_per_partition,
                                                           name=f"resnet_block_{res_block_idx}")
                                for res_block_idx in range(self.num_res_blocks)]

        # Likelihood distribution
        self.likelihood_dist = None

        # Likelihood of the most recent sample
        self.log_likelihood = -np.inf

        # this variable will allow us to perform Empirical Bayes on the first prior
        # Referred to as "h_top" in both the Kingma and Townsend implementations
        self._generative_base = tf.Variable(tf.zeros(self.deterministic_filters),
                                            name="generative_base")

        # ---------------------------------------------------------------------
        # EMA shadow variables
        # ---------------------------------------------------------------------
        self._ema_shadow_variables = {}

    def generative_base(self, batch_size, height, width):

        base = tf.reshape(self._generative_base, [1, 1, 1, self.deterministic_filters])

        return tf.tile(base, [batch_size, height // 2, width // 2, 1])

    @property
    def likelihood_function(self):

        likelihood_scale = tf.math.exp(self.likelihood_log_scale)

        def discretized_logistic(reference, reconstruction, binsize=1./256.):

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

        # TODO
        def discretized_laplace_log_prob(reference, reconstruction, binsize=1./256.):

            # Discretize the output
            discretized_input = tf.math.floor(reference / binsize) * binsize

        def ms_ssim_pseudo_log_prob(reference, reconstruction):
            return 1. / likelihood_scale * tf.image.ssim_multiscale(reference / likelihood_scale,
                                                                    reconstruction / likelihood_scale,
                                                                    max_val=1.0)

        if self._likelihood_function == "discretized_logistic":
            return discretized_logistic

        elif self._likelihood_function == "gaussian":
            return gaussian_log_prob

        elif self._likelihood_function == "laplace":
            return laplace_log_prob

        elif self._likelihood_function == "ms-ssim":
            return ms_ssim_pseudo_log_prob

        else:
            raise NotImplementedError

    def call(self, tensor, binsize=1 / 256.0):
        input = tensor
        batch_size, height, width, _ = input.shape

        # ---------------------------------------------------------------------
        # Perform Inference Pass
        # ---------------------------------------------------------------------
        tensor = self.first_infer_conv(tensor)

        # We go through the residual blocks in reverse topological order for the inference pass
        for res_block in self.residual_blocks[::-1]:
            tensor = res_block(tensor, inference_pass=True)

        # ---------------------------------------------------------------------
        # Perform Generative Pass
        # ---------------------------------------------------------------------
        tensor = self.generative_base(batch_size, height, width)

        # We go through the residual blocks in topological order for the generative pass
        for res_block in self.residual_blocks:
            tensor = res_block(tensor, inference_pass=False)

        reconstruction = tf.nn.elu(tensor)
        reconstruction = self.last_gen_conv(reconstruction)
        reconstruction = tf.clip_by_value(reconstruction, -0.5 + 1. / 512., 0.5 - 1. / 512.)

        # Gaussian Likelihood
        # self.likelihood_dist = tfd.Normal(loc=tensor,
        #                                   scale=1.)
        #
        # self.log_likelihood = self.likelihood_dist.log_prob(original_tensor)

        # Discretized Logistic Likelihood
        log_likelihood = self.likelihood_function(input, reconstruction)
        self.log_likelihood = tf.reduce_mean(log_likelihood)

        # If it's the initialization round, create our EMA shadow variables
        if not self.is_ema_variables_initialized:
            self.create_ema_variables()

        return reconstruction + 0.5

    def kl_divergence(self, empirical=False, minimum_kl=0., reduce=True):

        if self.use_iaf and not empirical:
            raise ModelError("KL divergence cannot be computed analytically when"
                             "using IAFs as posterior!")

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
        self.call(images)

        for res_block in self.residual_blocks:
            res_block.update_coders()

    def compress(self, image, seed, update_sampler=False):
        batch_size, height, width, _ = image.shape
        tensor = image

        tensor = self.first_infer_conv(tensor)

        # We first calculate the inference statistics of the image.
        # Note that the ResNet blocks are ordered according to the order of a generative pass,
        # so we iterate the list in reverse
        for resnet_block in self.residual_blocks[::-1]:
            tensor = resnet_block(tensor, inference_pass=True,)

        # Once the inference pass is complete, we code each of the blocks sequentially
        tensor = self.generative_base(batch_size=batch_size,
                                      width=width,
                                      height=height)

        block_indices = []
        for resnet_block in self.residual_blocks:
            indices, tensor = resnet_block(tensor,
                                           inference_pass=False,
                                           encoder_args={"seed": seed, "update_sampler": update_sampler})

            block_indices.append(indices)

        reconstruction = tf.nn.elu(tensor)
        reconstruction = self.last_gen_conv(reconstruction)
        reconstruction = tf.clip_by_value(reconstruction, -0.5 + 1. / 512., 0.5 - 1. / 512.)

        # Discretized Logistic Likelihood
        log_likelihood = self.likelihood_function(image, reconstruction)
        self.log_likelihood = tf.reduce_mean(log_likelihood)

        return block_indices, reconstruction

    def get_codelength(self, compressed_codes):
        codelength = 0.
        for resnet_block, compressed_code in zip(self.residual_blocks, compressed_codes):
            codelength += resnet_block.coder.get_codelength(compressed_code)
        return codelength


    def decompress(self, compressed_codes, seed, lossless=True):

        # TODO
        batch_size, height, width, _ = 1, 16, 16, None

        tensor = self.generative_base()

        # We sequentially decode through the resnet blocks
        for resnet_block, compressed_code in zip(self.residual_blocks, compressed_codes):
            tensor = resnet_block(tensor, inference_pass=False, decoder_args={"seed": seed,
                                                                              "indices": compressed_codes})

        reconstruction = tf.nn.elu(tensor)
        reconstruction = self.last_gen_conv(reconstruction)
        reconstruction = tf.clip_by_value(reconstruction, -0.5 + 1. / 512., 0.5 - 1. / 512.)

        return reconstruction + 0.5
