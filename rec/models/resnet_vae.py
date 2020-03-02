import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from .custom_modules import ReparameterizedConv2D, ReparameterizedConv2DTranspose, AutoRegressiveMultiConv2D

# ReparameterizedConv2D = tf.keras.layers.Conv2D
# ReparameterizedConv2DTranspose = tf.keras.layers.Conv2DTranspose

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
                 stochastic_filters,
                 deterministic_filters,
                 kernel_size=(3, 3),
                 use_iaf=False,
                 is_last=False,
                 name="bidirectional_resnet_block",
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

        self.iaf_autoregressive_context = None
        self.iaf_posterior_multiconv = None

        self.empirical_kld = 0.

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
        kld = tf.reduce_sum(tf.reduce_sum(kld, axis=[1, 2]), axis=[0])

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
        # in the original implementation the dimensions within a single filter do not
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
            self.iaf_autoregressive_context = ReparameterizedConv2D(
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

    def call(self, tensor, latent_code=None, inference_pass=True, eps=1e-7):
        """

        :param tensor: data to be passed through the residual block
        :param latent_code: If not None during a generative pass, we omit the calculation of the posterior parameters.
        If not None during an inference pass, it doesn't modify the behaviour. Must be the same shape as tensor
        :param inference_pass:
        :return:
        """

        if latent_code is not None:
            assert latent_code.shape == tensor.shape

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

            # Calculate next set of deterministic feautres
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

            # If no latent code is provided, we need to create it
            if latent_code is None:
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
                prior_log_prob = self.prior.log_prob(latent_code)

                if self.use_iaf:
                    context = self.iaf_autoregressive_context(tensor)

                    iaf_mean, iaf_log_scale = self.iaf_posterior_multiconv(latent_code,
                                                                           context=context)

                    iaf_mean = 0.1 * iaf_mean
                    iaf_log_scale = 0.1 * iaf_log_scale

                    # Update latent code
                    latent_code = (latent_code - iaf_mean) / tf.exp(iaf_log_scale)

                    # Update posterior log probability with IAF's Jacobian logdet
                    post_log_prob = post_log_prob + iaf_log_scale

                self.empirical_kld = post_log_prob - prior_log_prob

            # Calculate next set of deterministic features for residual block
            tensor = self.gen_conv1(tensor)

            # Concatenate code and generative features. The channels are always the last axis
            tensor = tf.concat([tensor, latent_code], axis=-1)

            tensor = tf.nn.elu(tensor)

            tensor = self.gen_conv2(tensor)

        # Add residual connection. Scaling factor taken from
        # https://github.com/hilloc-submission/hilloc/blob/b89e9c983e3764798e7c6f81f5cfc1d11b349d96/experiments/rvae/model/__init__.py#L116
        tensor = input + 0.1 * tensor

        return tensor


class BidirectionalResNetVAE(tfk.Model):
    """
    Implements the bidirectional ResNetVAE as described in:
    D. P. Kingma, T. Salimans, R. Jozefowicz, X. Chen, I. Sutskever, and M. Welling.
    Improved variational inference with inverse autoregressive flow.
    In Advances in Neural Information ProcessingSystems (NIPS), 2016.
    """

    def __init__(self,
                 num_res_blocks,
                 first_kernel_size=(5, 5),
                 first_strides=(2, 2),
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 deterministic_filters=160,
                 stochastic_filters=32,
                 use_iaf=False,
                 latent_size="variable",
                 name="resnet_vae",
                 **kwargs):
        super().__init__(name=name,
                         **kwargs)

        # ---------------------------------------------------------------------
        # Assign hyperparamteres
        # ---------------------------------------------------------------------
        self.num_res_blocks = num_res_blocks

        self.first_kernel_size = first_kernel_size
        self.first_strides = first_strides

        self.kernel_size = kernel_size
        self.strides = strides
        self.stochastic_filters = stochastic_filters
        self.deterministic_filters = deterministic_filters

        self.use_iaf = use_iaf

        # ---------------------------------------------------------------------
        # Create parameters
        # ---------------------------------------------------------------------
        self.likelihood_log_scale = tf.Variable(0., name="likelihood_log_scale")

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
                                                           kernel_size=self.kernel_size,
                                                           is_last=res_block_idx == 0,  # Declare last residual block
                                                           use_iaf=self.use_iaf,
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

    def generative_base(self, batch_size, height, width):

        base = tf.reshape(self._generative_base, [1, 1, 1, self.deterministic_filters])

        return tf.tile(base, [batch_size, height // 2, width // 2, 1])

    def call(self, tensor, binsize=1 / 256.0):
        input = tensor
        batch_size, height, width, _ = input.shape

        # ---------------------------------------------------------------------
        # Perform Inference Pass
        # ---------------------------------------------------------------------
        tensor = self.first_infer_conv(tensor)

        # We go through the residual blocks in reverse topological order for the inference pass
        for res_block in self.residual_blocks[::-1]:
            tensor = res_block(tensor, latent_code=None, inference_pass=True)

        # ---------------------------------------------------------------------
        # Perform Generative Pass
        # ---------------------------------------------------------------------
        tensor = self.generative_base(batch_size, height, width)

        # We go through the residual blocks in topological order for the generative pass
        for res_block in self.residual_blocks:
            tensor = res_block(tensor, latent_code=None, inference_pass=False)

        reconstruction = tf.nn.elu(tensor)
        reconstruction = self.last_gen_conv(reconstruction)
        reconstruction = tf.clip_by_value(reconstruction, -0.5 + 1. / 512., 0.5 - 1. / 512.)

        # Gaussian Likelihood
        # self.likelihood_dist = tfd.Normal(loc=tensor,
        #                                   scale=1.)
        #
        # self.log_likelihood = self.likelihood_dist.log_prob(original_tensor)

        # Discretized Logistic Likelihood
        likelihood_scale = tf.math.exp(self.likelihood_log_scale)

        # Discretize the output
        discretized_input = tf.math.floor(input / binsize) * binsize
        discretized_input = (discretized_input - reconstruction) / likelihood_scale

        self.log_likelihood = tf.nn.sigmoid(discretized_input + binsize / likelihood_scale)
        self.log_likelihood = self.log_likelihood - tf.nn.sigmoid(discretized_input)

        self.log_likelihood = tf.math.log(self.log_likelihood + 1e-7)
        self.log_likelihood = tf.reduce_mean(tf.reduce_sum(self.log_likelihood, [1, 2, 3]))

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
