import tensorflow as tf
import tensorflow_probability as tfp

from rec.models.custom_modules import SignalConv2D, GDN
from rec.models.lossy import LossyCompressionModel
from rec.io.utils import write_compressed_code, read_compressed_code

tfd = tfp.distributions
tfl = tf.keras.layers


class AnalysisTransform(tfl.Layer):

    def __init__(self,
                 num_filters,
                 name="analysis_transform",
                 **kwargs):
        super().__init__(name=name,
                         **kwargs)

        self.num_filters = num_filters

    def build(self, input_shape):
        self.layers = [
            SignalConv2D(filters=self.num_filters,
                         kernel=(9, 9),
                         name="conv_0",
                         corr=True,
                         strides_down=4,
                         padding="reflect",
                         use_bias=True),
            GDN(inverse=False,
                name="gdn_0"),
            SignalConv2D(filters=self.num_filters,
                         kernel=(5, 5),
                         name="conv_1",
                         corr=True,
                         strides_down=2,
                         padding="reflect",
                         use_bias=True),
            GDN(inverse=False,
                name="gdn_1"),
        ]

        self._posterior_loc_head = SignalConv2D(filters=self.num_filters,
                                                kernel=(5, 5),
                                                name="posterior_loc_head",
                                                corr=True,
                                                strides_down=2,
                                                padding="reflect",
                                                use_bias=False)

        self._posterior_log_scale_head = SignalConv2D(filters=self.num_filters,
                                                      kernel=(5, 5),
                                                      name="posterior_log_scale_head",
                                                      corr=True,
                                                      strides_down=2,
                                                      padding="reflect",
                                                      use_bias=False)

        super().build(input_shape)

    def call(self, tensor):
        for layer in self.layers:
            tensor = layer(tensor)

        posterior_loc = self._posterior_loc_head(tensor)
        posterior_scale = tf.nn.softplus(self._posterior_log_scale_head(tensor))

        return posterior_loc, posterior_scale


class SynthesisTransform(tfl.Layer):

    def __init__(self, num_filters, name="synthesis_transform", **kwargs):
        super().__init__(name=name, **kwargs)

        self.num_filters = num_filters

    def build(self, input_shape):
        self.layers = [
            SignalConv2D(filters=self.num_filters,
                         kernel=(5, 5),
                         name="conv_0",
                         corr=False,
                         strides_up=2,
                         padding="reflect",
                         use_bias=True),
            GDN(name="igdn_0",
                inverse=True),
            SignalConv2D(filters=self.num_filters,
                         kernel=(5, 5),
                         name="conv_1",
                         corr=False,
                         strides_up=2,
                         padding="reflect",
                         use_bias=True),
            GDN(name="igdn_1",
                inverse=True),
            SignalConv2D(filters=3,
                         kernel=(9, 9),
                         name="conv_2",
                         corr=False,
                         strides_up=4,
                         padding="reflect",
                         use_bias=True),
        ]

        super().build(input_shape)

    def call(self, tensor):
        for layer in self.layers:
            tensor = layer(tensor)

        return tensor


class Large1LevelVAE(LossyCompressionModel):

    def __init__(self, num_filters, name="large_1_level_vae", **kwargs):
        super().__init__(name=name, **kwargs)

        self.num_filters = num_filters

        self._prior_base = tf.Variable(tf.zeros((1, 1, 1, self.num_filters)), name="prior_base")

        self._prior_conv = SignalConv2D(filters=self.num_filters,
                                        kernel=(3, 3),
                                        corr=True,
                                        strides_down=1,
                                        padding="reflect",
                                        use_bias=True,
                                        name="prior_conv")

        self._prior_loc_head = SignalConv2D(filters=self.num_filters,
                                            kernel=(3, 3),
                                            corr=True,
                                            strides_down=1,
                                            padding="reflect",
                                            use_bias=True,
                                            name="prior_loc_head")

        self._prior_log_scale_head = SignalConv2D(filters=self.num_filters,
                                                  kernel=(3, 3),
                                                  corr=True,
                                                  strides_down=1,
                                                  padding="reflect",
                                                  use_bias=True,
                                                  name="prior_log_scale_head")

        self.analysis_transform = AnalysisTransform(num_filters=self.num_filters)
        self.synthesis_transform = SynthesisTransform(num_filters=self.num_filters)

    def prior_base(self, batch_size, height, width):
        return tf.tile(self._prior_base, [batch_size, height // 16, width // 16, 1])

    def kl_divergence(self):
        return [tf.reduce_mean(tf.reduce_sum(tfd.kl_divergence(self.posterior, self.prior), axis=[1, 2, 3]))]

    def call(self, tensor):

        inputs = tensor
        batch_size, height, width, _ = inputs.shape

        # =====================================================================
        # Inference Pass
        # =====================================================================
        posterior_loc, posterior_scale = self.analysis_transform(tensor)

        self.posterior = tfd.Normal(loc=posterior_loc,
                                    scale=posterior_scale)

        # =====================================================================
        # Generative Pass
        # =====================================================================

        # Calculate the prior
        tensor = self.prior_base(batch_size, height, width)
        tensor = self._prior_conv(tensor)
        tensor = tf.nn.elu(tensor)

        prior_loc = self._prior_loc_head(tensor)
        prior_scale = tf.nn.softplus(self._prior_log_scale_head(tensor))

        self.prior = tfd.Normal(loc=prior_loc,
                                scale=prior_scale)

        # Generate latent code from the posterior: z ~ q(z | x)
        tensor = self.posterior.sample()

        # Reconstruct image
        tensor = self.synthesis_transform(tensor)

        return tensor


