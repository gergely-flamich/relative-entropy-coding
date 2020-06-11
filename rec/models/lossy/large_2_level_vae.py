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
                         kernel=(5, 5),
                         name="conv_0",
                         corr=True,
                         strides_down=2,
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
            SignalConv2D(filters=self.num_filters,
                         kernel=(5, 5),
                         name="conv_2",
                         corr=True,
                         strides_down=2,
                         padding="reflect",
                         use_bias=True),
            GDN(inverse=False,
                name="gdn_2"),
        ]

        self._posterior_loc_head = SignalConv2D(filters=self.num_filters,
                                                kernel=(5, 5),
                                                name="posterior_loc_head",
                                                corr=True,
                                                strides_down=2,
                                                padding="reflect",
                                                use_bias=True)

        self._posterior_log_scale_head = SignalConv2D(filters=self.num_filters,
                                                      kernel=(5, 5),
                                                      name="posterior_log_scale_head",
                                                      corr=True,
                                                      strides_down=2,
                                                      padding="reflect",
                                                      use_bias=True)

        super().build(input_shape)

    def call(self, tensor):
        for layer in self.layers:
            tensor = layer(tensor)

        posterior_loc = self._posterior_loc_head(tensor)
        posterior_log_scale = self._posterior_log_scale_head(tensor)

        return posterior_loc, posterior_log_scale


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
            SignalConv2D(filters=self.num_filters,
                         kernel=(5, 5),
                         name="conv_2",
                         corr=False,
                         strides_up=2,
                         padding="reflect",
                         use_bias=True),
            GDN(name="igdn_2",
                inverse=True),
            SignalConv2D(filters=3,
                         kernel=(5, 5),
                         name="conv_3",
                         corr=False,
                         strides_up=2,
                         padding="reflect",
                         use_bias=True),
        ]

        super().build(input_shape)

    def call(self, tensor):
        for layer in self.layers:
            tensor = layer(tensor)

        return tensor


class HyperAnalysisTransform(tfl.Layer):

    def __init__(self, num_filters, name="hyper_analysis_transform", **kwargs):
        super().__init__(name=name,
                         **kwargs)

        self.num_filters = num_filters

    def build(self, input_shape):
        self.layers = [
            SignalConv2D(filters=self.num_filters,
                         kernel=(3, 3),
                         name="conv_0",
                         corr=True,
                         strides_down=1,
                         padding="reflect",
                         use_bias=True),
            tf.nn.relu,
            SignalConv2D(filters=self.num_filters,
                         kernel=(5, 5),
                         name="conv_1",
                         corr=True,
                         strides_down=2,
                         padding="reflect",
                         use_bias=True),
            tf.nn.relu,
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
        posterior_log_scale = self._posterior_log_scale_head(tensor)

        return posterior_loc, posterior_log_scale


class HyperSynthesisTransform(tfl.Layer):

    def __init__(self, num_filters, num_output_filters, name="hyper_synthesis_transform", **kwargs):
        super().__init__(name=name,
                         **kwargs)

        self.num_filters = num_filters
        self.num_output_filters = num_output_filters

    def build(self, input_shape):
        self.layers = [
            SignalConv2D(filters=self.num_filters,
                         kernel=(5, 5),
                         name="conv_0",
                         corr=False,
                         strides_up=2,
                         padding="reflect",
                         use_bias=True,
                         dft_kernel_parametrization=False),
            tf.nn.relu,
            SignalConv2D(filters=self.num_filters,
                         kernel=(5, 5),
                         name="conv_1",
                         corr=False,
                         strides_up=2,
                         padding="reflect",
                         use_bias=True,
                         dft_kernel_parametrization=False),
            tf.nn.relu,

        ]

        self._prior_loc_head = SignalConv2D(filters=self.num_output_filters,
                                            kernel=(3, 3),
                                            name="prior_loc_head",
                                            corr=False,
                                            strides_up=1,
                                            padding="reflect",
                                            use_bias=True,
                                            dft_kernel_parametrization=False)

        self._prior_log_scale_head = SignalConv2D(filters=self.num_output_filters,
                                                  kernel=(3, 3),
                                                  name="prior_log_scale_head",
                                                  corr=False,
                                                  strides_up=1,
                                                  padding="reflect",
                                                  use_bias=True,
                                                  dft_kernel_parametrization=False)

        super().build(input_shape)

    def call(self, tensor):
        for layer in self.layers:
            tensor = layer(tensor)

        prior_loc = self._prior_loc_head(tensor)
        prior_log_scale = self._prior_log_scale_head(tensor)

        return prior_loc, prior_log_scale


class Large2LevelVAE(LossyCompressionModel):

    def __init__(self, level_1_filters, level_2_filters, name="large_2_level_vae", **kwargs):
        super().__init__(name=name, **kwargs)

        self.level_1_filters = level_1_filters
        self.level_2_filters = level_2_filters

        self._prior_base = tf.Variable(tf.zeros((1, 1, 1, self.level_2_filters)), name="prior_base")

        self._prior_conv = SignalConv2D(filters=self.level_2_filters,
                                        kernel=(3, 3),
                                        corr=True,
                                        strides_down=1,
                                        padding="reflect",
                                        use_bias=True,
                                        name="level_2_prior_conv")

        self._prior_loc_head = SignalConv2D(filters=self.level_2_filters,
                                            kernel=(3, 3),
                                            corr=True,
                                            strides_down=1,
                                            padding="reflect",
                                            use_bias=True,
                                            name="level_2_prior_loc_head")

        self._prior_log_scale_head = SignalConv2D(filters=self.level_2_filters,
                                                  kernel=(3, 3),
                                                  corr=True,
                                                  strides_down=1,
                                                  padding="reflect",
                                                  use_bias=True,
                                                  name="level_2_prior_log_scale_head")

        # 1 x 1 convolutions that will combine the inference posterior statistics with
        # the generative prior statistics to get the level 1 posterior statistics
        self._level_1_posterior_loc_combiner = tfl.Conv2D(filters=self.level_1_filters,
                                                          kernel_size=(1, 1),
                                                          strides=1,
                                                          padding="valid",
                                                          use_bias=True,
                                                          name="level_1_posterior_loc_combiner")

        self._level_1_posterior_log_scale_combiner = tfl.Conv2D(filters=self.level_1_filters,
                                                                kernel_size=(1, 1),
                                                                strides=1,
                                                                padding="valid",
                                                                use_bias=True,
                                                                name="level_1_posterior_log_scale_combiner")

        self.analysis_transform = AnalysisTransform(num_filters=self.level_1_filters)
        self.synthesis_transform = SynthesisTransform(num_filters=self.level_1_filters)

        self.hyper_analysis_transform = HyperAnalysisTransform(num_filters=self.level_2_filters)
        self.hyper_synthesis_transform = HyperSynthesisTransform(num_filters=self.level_2_filters,
                                                                 num_output_filters=self.level_1_filters)

    def prior_base(self, batch_size, height, width):
        return tf.tile(self._prior_base, [batch_size, height // 64, width // 64, 1])

    def kl_divergence(self):
        return [tf.reduce_mean(tf.reduce_sum(tfd.kl_divergence(posterior, prior), axis=[1, 2, 3]))
                for posterior, prior in [(self.level_1_posterior, self.level_1_prior),
                                         (self.level_2_posterior, self.level_2_prior)]]

    def call(self, tensor, sampling_fn=None):
        inputs = tensor
        batch_size, height, width, _ = inputs.shape

        # =====================================================================
        # Inference Pass
        # =====================================================================

        # Get level 1 inference statistics
        level_1_posterior_loc, level_1_posterior_log_scale = self.analysis_transform(tensor)

        # Pass on the posterior location
        level_2_posterior_loc, level_2_posterior_log_scale = self.hyper_analysis_transform(level_1_posterior_loc)
        level_2_posterior_scale = tf.nn.softplus(level_2_posterior_log_scale) + 1e-7

        # Get level 2 inference statistics
        self.level_2_posterior = tfd.Normal(loc=level_2_posterior_loc,
                                            scale=level_2_posterior_scale)

        # =====================================================================
        # Generative Pass
        # =====================================================================

        # Calculate the empirical Bayes prior on level 2
        tensor = self.prior_base(batch_size, height, width)
        tensor = self._prior_conv(tensor)
        tensor = tf.nn.elu(tensor)

        level_2_prior_loc = self._prior_loc_head(tensor)
        level_2_prior_scale = tf.nn.softplus(self._prior_log_scale_head(tensor)) + 1e-7

        self.level_2_prior = tfd.Normal(loc=level_2_prior_loc,
                                        scale=level_2_prior_scale)

        # Generate latent code from the hyper posterior: z ~ q(z | x)
        if sampling_fn is None:
            tensor = self.level_2_posterior.sample()

        else:
            level_2_indices, tensor = sampling_fn(target=self.level_2_posterior,
                                                  coder=self.level_2_prior)

        # Calculate level 1 prior statistics
        level_1_prior_loc, level_1_prior_log_scale = self.hyper_synthesis_transform(tensor)
        level_1_prior_scale = tf.nn.softplus(level_1_prior_log_scale) + 1e-7

        # --------------------------------------------------------------
        # Combine level 1 prior and level 1 inference posterior statistics
        # --------------------------------------------------------------

        # Concatenate along the channels
        level_1_posterior_loc = tf.concat([level_1_posterior_loc, level_1_prior_loc], axis=-1)
        level_1_posterior_log_scale = tf.concat([level_1_posterior_log_scale, level_1_prior_log_scale], axis=-1)

        level_1_posterior_loc = tf.nn.elu(level_1_posterior_loc)
        level_1_posterior_log_scale = tf.nn.elu(level_1_posterior_log_scale)

        # Combine the statistics using 1 x 1 convolutions
        level_1_posterior_loc = self._level_1_posterior_loc_combiner(level_1_posterior_loc)
        level_1_posterior_log_scale = self._level_1_posterior_log_scale_combiner(level_1_posterior_log_scale)
        level_1_posterior_scale = tf.nn.softplus(level_1_posterior_log_scale) + 1e-7

        # Create level 1 latent distributions
        self.level_1_prior = tfd.Normal(loc=level_1_prior_loc,
                                        scale=level_1_prior_scale)

        self.level_1_posterior = tfd.Normal(loc=level_1_posterior_loc,
                                            scale=level_1_posterior_scale)

        # Generate latent code from the level 1 posterior: y ~ q(y | z, x)
        if sampling_fn is None:
            tensor = self.level_1_posterior.sample()

        else:
            level_1_indices, tensor = sampling_fn(target=self.level_1_posterior,
                                                  coder=self.level_1_prior)

        # Reconstruct image
        tensor = self.synthesis_transform(tensor)

        if sampling_fn is None:
            return tensor

        else:
            return [level_2_indices, level_1_indices], tensor

    def compress(self, file_path, image, seed, sampler, block_size, max_index):

        sampling_fn = lambda target, coder: sampler.encode(target, coder, seed=seed)

        block_indices, reconstruction = self(image[None, ...], sampling_fn=sampling_fn)

        write_compressed_code(file_path=file_path,
                              seed=seed,
                              image_shape=image.shape,
                              block_size=block_size,
                              block_indices=block_indices,
                              max_index=max_index)

        return reconstruction

    def decompress(self, file_path, sampler):

        # Recover stuff from the binary representation
        seed, image_shape, block_size, block_indices = read_compressed_code(file_path=file_path)
        batch_size, height, width = image_shape

        # Recover image from a generative pass

        # Calculate the empirical Bayes prior on level 2
        tensor = self.prior_base(batch_size, height, width)
        tensor = self._prior_conv(tensor)
        tensor = tf.nn.elu(tensor)

        level_2_prior_loc = self._prior_loc_head(tensor)
        level_2_prior_scale = tf.nn.softplus(self._prior_log_scale_head(tensor)) + 1e-7

        self.level_2_prior = tfd.Normal(loc=level_2_prior_loc,
                                        scale=level_2_prior_scale)

        # Decode level 2 sample
        tensor = sampler.decode(self.level_2_prior, seed=seed, indices=block_indices[0])

        # Calculate level 1 prior statistics
        level_1_prior_loc, level_1_prior_log_scale = self.hyper_synthesis_transform(tensor)
        level_1_prior_scale = tf.nn.softplus(level_1_prior_log_scale) + 1e-7

        self.level_1_prior = tfd.Normal(loc=level_1_prior_loc,
                                        scale=level_1_prior_scale)

        # Decode level 1 sample
        tensor = sampler.decode(self.level_1_prior, seed=seed, indices=block_indices[1])

        # Reconstruct image
        tensor = self.synthesis_transform(tensor)

        return tensor
