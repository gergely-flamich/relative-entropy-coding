import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

tfl = tf.keras.layers
tfk = tf.keras
tfd = tfp.distributions


class BidirectionalResidualBlock(tfl.Layer):
    """
    Implements a bidirectional Resnet Block
    """

    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
                 name="bidirectional_resnet_block",
                 **kwargs):
        super().__init__(name=name,
                         **kwargs)

        self.filters = filters
        self.kernel_size = kernel_size

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
        self.infer_posterior_scale = 1.

        self.gen_posterior_loc = 0.
        self.gen_posterior_scale = 1.

        self.prior_loc = 0.
        self.prior_scale = 1.

        # ---------------------------------------------------------------------
        # Distributions associated with the current residual block
        # ---------------------------------------------------------------------
        self.posterior = None
        self.prior = None

    @property
    def posterior_loc(self):
        return self.infer_posterior_loc + self.gen_posterior_loc

    @property
    def posterior_scale(self):
        return self.infer_posterior_scale + self.gen_posterior_scale

    def build(self, input_shape):
        # ---------------------------------------------------------------------
        # Stuff for the inference side
        # ---------------------------------------------------------------------
        self.infer_conv1 = tfl.Conv2D(filters=self.filters,
                                      kernel_size=self.kernel_size,
                                      strides=(1, 1),
                                      padding="same")

        self.infer_conv2 = tfl.Conv2D(filters=self.filters,
                                      kernel_size=self.kernel_size,
                                      strides=(1, 1),
                                      padding="same")

        self.infer_posterior_loc_head = tfl.Conv2D(filters=self.filters,
                                                   kernel_size=self.kernel_size,
                                                   strides=(1, 1),
                                                   padding="same")

        self.infer_posterior_log_scale_head = tfl.Conv2D(filters=self.filters,
                                                         kernel_size=self.kernel_size,
                                                         strides=(1, 1),
                                                         padding="same")

        # ---------------------------------------------------------------------
        # Stuff for the generative side
        # ---------------------------------------------------------------------
        self.gen_conv1 = tfl.Conv2DTranspose(filters=self.filters,
                                             kernel_size=self.kernel_size,
                                             strides=(1, 1),
                                             padding="same")

        self.gen_conv2 = tfl.Conv2DTranspose(filters=self.filters,
                                             kernel_size=self.kernel_size,
                                             strides=(1, 1),
                                             padding="same")

        self.prior_loc_head = tfl.Conv2DTranspose(filters=self.filters,
                                                  kernel_size=self.kernel_size,
                                                  strides=(1, 1),
                                                  padding="same")

        self.prior_log_scale_head = tfl.Conv2DTranspose(filters=self.filters,
                                                        kernel_size=self.kernel_size,
                                                        strides=(1, 1),
                                                        padding="same")

        self.gen_posterior_loc_head = tfl.Conv2DTranspose(filters=self.filters,
                                                          kernel_size=self.kernel_size,
                                                          strides=(1, 1),
                                                          padding="same")

        self.gen_posterior_log_scale_head = tfl.Conv2DTranspose(filters=self.filters,
                                                                kernel_size=self.kernel_size,
                                                                strides=(1, 1),
                                                                padding="same")

        super().build(input_shape=input_shape)

    def call(self, tensor, latent_code=None, inference_pass=True):
        """

        :param tensor: data to be passed through the residual block
        :param latent_code: If not None during a generative pass, we omit the calculation of the posterior parameters.
        If not None during an inference pass, it doesn't modify the behaviour. Must be the same shape as tensor
        :param inference_pass:
        :return:
        """

        if latent_code is not None:
            assert latent_code.shape == tensor.shape

        original_tensor = tensor

        # First layer in block
        tensor = tf.nn.elu(tensor)

        # ---------------------------------------------------------------------
        # Inference pass
        # ---------------------------------------------------------------------
        if inference_pass:

            # Calculate first part of posterior statistics
            self.infer_posterior_loc = self.infer_posterior_loc_head(tensor)
            self.infer_posterior_scale = tf.math.exp(self.infer_posterior_log_scale_head(tensor))

            # Calculate next set of deterministic feautres
            tensor = self.infer_conv1(tensor)
            tensor = tf.nn.elu(tensor)

            tensor = self.infer_conv2(tensor)

        # ---------------------------------------------------------------------
        # Generative pass
        # ---------------------------------------------------------------------
        else:
            # If no latent code is provided, we need to create it
            if latent_code is None:
                # Calculate second part of posterior statistics
                self.gen_posterior_loc = self.gen_posterior_loc_head(tensor)
                self.gen_posterior_scale = tf.math.exp(self.gen_posterior_log_scale_head(tensor))

                # Sample from posterior
                self.posterior = tfd.Normal(loc=self.posterior_loc,
                                            scale=self.posterior_scale)

                latent_code = self.posterior.sample()

            # Calculate prior parameters
            self.prior_loc = self.prior_loc_head(tensor)
            self.prior_scale = tf.math.exp(self.prior_log_scale_head(tensor))

            self.prior = tfd.Normal(loc=self.prior_loc,
                                    scale=self.prior_scale)

            # Calculate next set of deterministic features for residual block
            tensor = self.gen_conv1(tensor)
            tensor = tf.nn.elu(tensor)

            # Concatenate code and generative features. The channels are always the last axis
            tensor = tf.concat([tensor, latent_code], axis=-1)
            tensor = self.gen_conv2(tensor)

        # Add residual connection
        tensor = original_tensor + tensor

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
                 first_filters=32,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 filters=32,
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
        self.first_filters = first_filters

        self.kernel_size = kernel_size
        self.strides = strides
        self.filters = filters

        # ---------------------------------------------------------------------
        # Create parameters
        # ---------------------------------------------------------------------
        self.likelihood_log_scale = tf.Variable(0., name="likelihood_log_scale")

        # ---------------------------------------------------------------------
        # Create ResNet Layers
        # ---------------------------------------------------------------------
        self.first_infer_conv = tfl.Conv2D(kernel_size=self.first_kernel_size,
                                           strides=self.first_strides,
                                           filters=self.first_filters,
                                           padding="same")

        self.last_gen_conv = tfl.Conv2DTranspose(kernel_size=self.first_kernel_size,
                                                 strides=self.first_strides,
                                                 filters=3,
                                                 padding="same")

        # We create these in topological order.
        # This means that residual_blocks[0] will have the bottom-most stochastic layer
        # And residual_blocks[-1] will have the top-most one, the output of which should be passed to last_gen_conv
        self.residual_blocks = [BidirectionalResidualBlock(filters=self.filters,
                                                           kernel_size=self.kernel_size)
                                for _ in range(self.num_res_blocks)]

        # Likelihood distribution
        self.likelihood_dist = None

        # Likelihood of the most recent sample
        self.log_likelihood = -np.inf

    def call(self, tensor, binsize=1/256.):
        original_tensor = tensor

        # Center tensor
        tensor = tensor - 0.5

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

        # We go through the residual blocks in topological order for the generative pass
        for res_block in self.residual_blocks[::-1]:
            tensor = res_block(tensor, latent_code=None, inference_pass=False)

        tensor = tf.nn.elu(tensor)
        tensor = self.last_gen_conv(tensor)

        # Note: This scaling is taken from the original implementation:
        # https://github.com/openai/iaf/blob/ad33fe4872bf6e4b4f387e709a625376bb8b0d9d/models.py#L468
        tensor = 0.1 * tensor + 0.5

        # Gaussian Likelihood
        # self.likelihood_dist = tfd.Normal(loc=tensor,
        #                                   scale=1.)
        #
        # self.log_likelihood = self.likelihood_dist.log_prob(original_tensor)

        # Discretized Logistic Likelihood
        tensor = tf.clip_by_value(tensor, 1. / 512., 1. - 1. / 512.)

        likelihood_scale = tf.math.exp(self.likelihood_log_scale)

        # Discretize the output
        original_tensor = tf.math.floor(original_tensor / binsize) * binsize
        original_tensor = (original_tensor - tensor) / likelihood_scale

        self.log_likelihood = tf.nn.sigmoid(original_tensor + binsize / likelihood_scale) - tf.nn.sigmoid(original_tensor)
        self.log_likelihood = tf.math.log(self.log_likelihood + 1e-7)

        return tensor

    def kl_divergence(self, empirical=False):

        if empirical:
            empirical_kls = []
            for res_block in self.residual_blocks:
                posterior_sample = res_block.posterior.sample()
                emp_kl = res_block.posterior.log_prob(posterior_sample) - res_block.prior.log_prob(posterior_sample)

                empirical_kls.append(emp_kl)

            return empirical_kls
        else:
            return [tfd.kl_divergence(res_block.posterior, res_block.prior) for res_block in self.residual_blocks]
