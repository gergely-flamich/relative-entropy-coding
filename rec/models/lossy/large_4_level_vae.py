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
                         use_bias=True,
                         activation=GDN(inverse=False,
                                        name="gdn_0"),
                         ),
            SignalConv2D(filters=self.num_filters,
                         kernel=(5, 5),
                         name="conv_1",
                         corr=True,
                         strides_down=2,
                         padding="reflect",
                         use_bias=True,
                         activation=GDN(inverse=False,
                                        name="gdn_1")),
            SignalConv2D(filters=self.num_filters,
                         kernel=(5, 5),
                         name="conv_2",
                         corr=True,
                         strides_down=2,
                         padding="reflect",
                         use_bias=True,
                         activation=GDN(inverse=False,
                                        name="gdn_2")),
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

        self._deterministic_features_head = SignalConv2D(filters=self.num_filters,
                                                         kernel=(5, 5),
                                                         name="deterministic_feautres_head",
                                                         corr=True,
                                                         strides_down=2,
                                                         padding="reflect",
                                                         use_bias=True)

        super().build(input_shape)

    def call(self, tensor):
        first_level_features = self.layers[0](tensor)

        tensor = first_level_features
        for layer in self.layers[1:]:
            tensor = layer(tensor)

        posterior_loc = self._posterior_loc_head(tensor)
        posterior_log_scale = self._posterior_log_scale_head(tensor)
        features = self._deterministic_features_head(tensor)

        return posterior_loc, posterior_log_scale, features, first_level_features


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
                         use_bias=True,
                         activation=GDN(name="igdn_0",
                                        inverse=True)),
            SignalConv2D(filters=self.num_filters,
                         kernel=(5, 5),
                         name="conv_1",
                         corr=False,
                         strides_up=2,
                         padding="reflect",
                         use_bias=True,
                         activation=GDN(name="igdn_1",
                                        inverse=True)),
            SignalConv2D(filters=self.num_filters,
                         kernel=(5, 5),
                         name="conv_2",
                         corr=False,
                         strides_up=2,
                         padding="reflect",
                         use_bias=True,
                         activation=GDN(name="igdn_2",
                                        inverse=True)),
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


class ExtendedAnalysisTransform(tfl.Layer):

    def __init__(self, num_filters, name="extended_analysis_transform", **kwargs):
        super().__init__(name=name,
                         **kwargs)

        self.num_filters = num_filters

    def build(self, input_shape):
        self.layers = [
            SignalConv2D(filters=self.num_filters,
                         kernel=(3, 3),
                         corr=True,
                         strides_down=1,
                         padding="reflect",
                         use_bias=True,
                         name="conv_0",
                         activation=GDN(inverse=False,
                                        name="gdn_0")),
            SignalConv2D(filters=self.num_filters,
                         kernel=(3, 3),
                         corr=True,
                         strides_down=1,
                         padding="reflect",
                         use_bias=True,
                         name="conv_1",
                         activation=GDN(inverse=False,
                                        name="gdn_1")),
        ]

        self._posterior_loc_head = SignalConv2D(filters=self.num_filters,
                                                kernel=(3, 3),
                                                name="posterior_loc_head",
                                                corr=True,
                                                strides_down=1,
                                                padding="reflect",
                                                use_bias=True)

        self._posterior_log_scale_head = SignalConv2D(filters=self.num_filters,
                                                      kernel=(3, 3),
                                                      name="posterior_log_scale_head",
                                                      corr=True,
                                                      strides_down=1,
                                                      padding="reflect",
                                                      use_bias=True)

        self._deterministic_features_head = SignalConv2D(filters=self.num_filters,
                                                         kernel=(3, 3),
                                                         name="deterministic_feautres_head",
                                                         corr=True,
                                                         strides_down=1,
                                                         padding="reflect",
                                                         use_bias=True)

        super().build(input_shape)

    def call(self, tensor):
        for layer in self.layers:
            tensor = layer(tensor)

        posterior_loc = self._posterior_loc_head(tensor)
        posterior_log_scale = self._posterior_log_scale_head(tensor)
        features = self._deterministic_features_head(tensor)

        return posterior_loc, posterior_log_scale, features


class ExtendedSynthesisTransform(tfl.Layer):

    def __init__(self, num_filters,
                 num_output_filters,
                 name="extended_synthesis_transform",
                 **kwargs):
        super().__init__(name=name,
                         **kwargs)

        self.num_filters = num_filters
        self.num_output_filters = num_output_filters

    def build(self, input_shape):
        self.layers = [
            SignalConv2D(filters=self.num_filters,
                         kernel=(3, 3),
                         corr=False,
                         strides_up=1,
                         padding="reflect",
                         use_bias=True,
                         name="conv_0",
                         activation=GDN(inverse=True,
                                        name="gdn_0")),
            SignalConv2D(filters=self.num_filters,
                         kernel=(3, 3),
                         corr=False,
                         strides_up=1,
                         padding="reflect",
                         use_bias=True,
                         name="conv_1",
                         activation=GDN(inverse=True)),
        ]

        self._prior_loc_head = SignalConv2D(filters=self.num_output_filters,
                                            kernel=(3, 3),
                                            name="prior_loc_head",
                                            corr=False,
                                            strides_up=1,
                                            padding="reflect",
                                            use_bias=True)

        self._prior_log_scale_head = SignalConv2D(filters=self.num_output_filters,
                                                  kernel=(3, 3),
                                                  name="prior_log_scale_head",
                                                  corr=False,
                                                  strides_up=1,
                                                  padding="reflect",
                                                  use_bias=True)

        self._deterministic_features_head = SignalConv2D(filters=self.num_output_filters,
                                                         kernel=(3, 3),
                                                         name="deterministic_feautres_head",
                                                         corr=False,
                                                         strides_up=1,
                                                         padding="reflect",
                                                         use_bias=True)

    def call(self, tensor):
        for layer in self.layers:
            tensor = layer(tensor)

        prior_loc = self._prior_loc_head(tensor)
        prior_log_scale = self._prior_log_scale_head(tensor)
        features = self._deterministic_features_head(tensor)

        return prior_loc, prior_log_scale, features


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
                         use_bias=True,
                         activation=tf.nn.elu),
            SignalConv2D(filters=self.num_filters,
                         kernel=(5, 5),
                         name="conv_1",
                         corr=True,
                         strides_down=2,
                         padding="reflect",
                         use_bias=True,
                         activation=tf.nn.elu),
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

        self._deterministic_features_head = SignalConv2D(filters=self.num_filters,
                                                         kernel=(5, 5),
                                                         name="deterministic_features_head",
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
        features = self._deterministic_features_head(tensor)

        return posterior_loc, posterior_log_scale, features


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
                         activation=tf.nn.elu),
            SignalConv2D(filters=self.num_filters,
                         kernel=(5, 5),
                         name="conv_1",
                         corr=False,
                         strides_up=2,
                         padding="reflect",
                         use_bias=True,
                         activation=tf.nn.elu),
        ]

        self._prior_loc_head = SignalConv2D(filters=self.num_output_filters,
                                            kernel=(3, 3),
                                            name="prior_loc_head",
                                            corr=False,
                                            strides_up=1,
                                            padding="reflect",
                                            use_bias=True)

        self._prior_log_scale_head = SignalConv2D(filters=self.num_output_filters,
                                                  kernel=(3, 3),
                                                  name="prior_log_scale_head",
                                                  corr=False,
                                                  strides_up=1,
                                                  padding="reflect",
                                                  use_bias=True)

        self._deterministic_features_head = SignalConv2D(filters=self.num_output_filters,
                                                         kernel=(3, 3),
                                                         name="deterministic_features_head",
                                                         corr=False,
                                                         strides_up=1,
                                                         padding="reflect",
                                                         use_bias=True)

        super().build(input_shape)

    def call(self, tensor):
        for layer in self.layers:
            tensor = layer(tensor)

        prior_loc = self._prior_loc_head(tensor)
        prior_log_scale = self._prior_log_scale_head(tensor)
        features = self._deterministic_features_head(tensor)

        return prior_loc, prior_log_scale, features


class ExtendedHyperAnalysisTransform(tfl.Layer):

    def __init__(self, num_filters,
                 name="extended_hyper_analysis_transform",
                 **kwargs):
        super().__init__(name=name,
                         **kwargs)

        self.num_filters = num_filters

    def build(self, input_shape):
        self.layers = [
            SignalConv2D(filters=self.num_filters,
                         kernel=(3, 3),
                         corr=True,
                         strides_down=1,
                         padding="reflect",
                         use_bias=True,
                         name="conv_0",
                         activation=tf.nn.elu),
            SignalConv2D(filters=self.num_filters,
                         kernel=(3, 3),
                         corr=True,
                         strides_down=1,
                         padding="reflect",
                         use_bias=True,
                         name="conv_1",
                         activation=tf.nn.elu),
        ]

        self._posterior_loc_head = SignalConv2D(filters=self.num_filters,
                                                kernel=(3, 3),
                                                name="posterior_loc_head",
                                                corr=True,
                                                strides_down=1,
                                                padding="reflect",
                                                use_bias=True)

        self._posterior_log_scale_head = SignalConv2D(filters=self.num_filters,
                                                      kernel=(3, 3),
                                                      name="posterior_log_scale_head",
                                                      corr=True,
                                                      strides_down=1,
                                                      padding="reflect",
                                                      use_bias=True)

        super().build(input_shape)

    def call(self, tensor):
        for layer in self.layers:
            tensor = layer(tensor)

        posterior_loc = self._posterior_loc_head(tensor)
        posterior_log_scale = self._posterior_log_scale_head(tensor)

        return posterior_loc, posterior_log_scale


class ExtendedHyperSynthesisTransform(tfl.Layer):

    def __init__(self,
                 num_filters,
                 num_output_filters,
                 name="extended_hyper_synthesis_transform",
                 **kwargs):
        super().__init__(name=name,
                         **kwargs)

        self.num_filters = num_filters
        self.num_output_filters = num_output_filters

    def build(self, input_shape):
        self.layers = [
            SignalConv2D(filters=self.num_filters,
                         kernel=(3, 3),
                         corr=False,
                         strides_up=1,
                         padding="reflect",
                         use_bias=True,
                         name="conv_0",
                         activation=tf.nn.elu),
            SignalConv2D(filters=self.num_filters,
                         kernel=(3, 3),
                         corr=False,
                         strides_up=1,
                         padding="reflect",
                         use_bias=True,
                         name="conv_1",
                         activation=tf.nn.elu),
        ]

        self._prior_loc_head = SignalConv2D(filters=self.num_output_filters,
                                            kernel=(3, 3),
                                            name="prior_loc_head",
                                            corr=False,
                                            strides_up=1,
                                            padding="reflect",
                                            use_bias=True)

        self._prior_log_scale_head = SignalConv2D(filters=self.num_output_filters,
                                                  kernel=(3, 3),
                                                  name="prior_log_scale_head",
                                                  corr=False,
                                                  strides_up=1,
                                                  padding="reflect",
                                                  use_bias=True)

        self._deterministic_features_head = SignalConv2D(filters=self.num_output_filters,
                                                         kernel=(3, 3),
                                                         name="deterministic_feautres_head",
                                                         corr=False,
                                                         strides_up=1,
                                                         padding="reflect",
                                                         use_bias=True)

    def call(self, tensor):
        for layer in self.layers:
            tensor = layer(tensor)

        prior_loc = self._prior_loc_head(tensor)
        prior_log_scale = self._prior_log_scale_head(tensor)
        features = self._deterministic_features_head(tensor)

        return prior_loc, prior_log_scale, features


class EmpiricalHyperPrior(tfl.Layer):

    def __init__(self,
                 num_filters,
                 name="empirical_hyper_prior",
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self.num_filters = num_filters

    def build(self, input_shape):
        self._prior_base = tf.Variable(tf.zeros((1, 1, 1, self.num_filters)), name="prior_base")

        self._prior_conv = SignalConv2D(filters=self.num_filters,
                                        kernel=(3, 3),
                                        corr=True,
                                        strides_down=1,
                                        padding="reflect",
                                        use_bias=True,
                                        name="conv_0")

        self._prior_loc_head = SignalConv2D(filters=self.num_filters,
                                            kernel=(3, 3),
                                            corr=True,
                                            strides_down=1,
                                            padding="reflect",
                                            use_bias=True,
                                            name="loc_head")

        self._prior_log_scale_head = SignalConv2D(filters=self.num_filters,
                                                  kernel=(3, 3),
                                                  corr=True,
                                                  strides_down=1,
                                                  padding="reflect",
                                                  use_bias=True,
                                                  name="log_scale_head")

        super().build(input_shape)

    def call(self, batch_size, height, width):
        tensor = tf.tile(self._prior_base, [batch_size, height, width, 1])

        tensor = self._prior_conv(tensor)
        tensor = tf.nn.elu(tensor)

        prior_loc = self._prior_loc_head(tensor)
        prior_log_scale = self._prior_log_scale_head(tensor)

        return prior_loc, prior_log_scale, tensor


class Large4LevelVAE(LossyCompressionModel):

    def __init__(self,
                 level_1_filters=192,
                 level_2_filters=192,
                 level_3_filters=128,
                 level_4_filters=128,
                 name="large_level_4_vae",
                 **kwargs):
        super().__init__(name=name,
                         **kwargs)

        self.level_1_filters = level_1_filters
        self.level_2_filters = level_2_filters
        self.level_3_filters = level_3_filters
        self.level_4_filters = level_4_filters

        # --------------------------------------------------------------
        # Define the main components of the model
        # --------------------------------------------------------------
        self.analysis_transform = AnalysisTransform(num_filters=level_1_filters)
        self.synthesis_transform = SynthesisTransform(num_filters=level_1_filters)

        self.extended_analysis_transform = ExtendedAnalysisTransform(num_filters=level_2_filters)
        self.extended_synthesis_transform = ExtendedSynthesisTransform(num_filters=level_2_filters,
                                                                       num_output_filters=level_1_filters)

        self.hyper_analysis_transform = HyperAnalysisTransform(num_filters=level_3_filters)
        self.hyper_synthesis_transform = HyperSynthesisTransform(num_filters=level_3_filters,
                                                                 num_output_filters=level_2_filters)

        self.extended_hyper_analysis_transform = ExtendedHyperAnalysisTransform(num_filters=level_4_filters)
        self.extended_hyper_synthesis_transform = ExtendedHyperSynthesisTransform(num_filters=level_4_filters,
                                                                                  num_output_filters=level_3_filters)

        self.empirical_hyper_prior = EmpiricalHyperPrior(num_filters=level_4_filters)

        # --------------------------------------------------------------
        # Define inference time residual connectors and combiners
        # --------------------------------------------------------------
        self.inputs_to_level_1_connector = SignalConv2D(filters=level_1_filters,
                                                        kernel=(9, 9),
                                                        strides_down=8,
                                                        corr=True,
                                                        padding="reflect",
                                                        use_bias=True,
                                                        name="inputs_to_level_1_connector")

        # We will compose with the inputs to level 1 connector
        self.inputs_to_level_2_connector = tfl.Conv2D(filters=level_2_filters,
                                                      kernel_size=(1, 1),
                                                      strides=1,
                                                      padding="valid",
                                                      use_bias=True,
                                                      name="inputs_to_level_2_connector")

        self.level_1_to_level_2_connector = tfl.Conv2D(filters=level_2_filters,
                                                       kernel_size=(1, 1),
                                                       strides=1,
                                                       padding="valid",
                                                       use_bias=True,
                                                       name="level_1_to_level_2_connector")

        # We will compose with the inputs to level 1 connector
        self.inputs_to_level_3_connector = SignalConv2D(filters=level_3_filters,
                                                        kernel=(5, 5),
                                                        strides_down=4,
                                                        corr=True,
                                                        padding="reflect",
                                                        use_bias=True,
                                                        name="inputs_to_level_3_connector")

        self.level_1_to_level_3_connector = SignalConv2D(filters=level_3_filters,
                                                         kernel=(5, 5),
                                                         strides_down=4,
                                                         corr=True,
                                                         padding="reflect",
                                                         use_bias=True,
                                                         name="level_1_to_level_3_connector")

        self.level_2_to_level_3_connector = SignalConv2D(filters=level_3_filters,
                                                         kernel=(5, 5),
                                                         strides_down=4,
                                                         corr=True,
                                                         padding="reflect",
                                                         use_bias=True,
                                                         name="level_2_to_level_3_connector")

        self.inference_combiners = [tfl.Conv2D(filters=filters,
                                               kernel_size=(1, 1),
                                               strides=1,
                                               padding="valid",
                                               use_bias=True,
                                               name=f"level_{i + 1}_inference_combiner")
                                    for i, filters in enumerate([level_1_filters,
                                                                 level_2_filters,
                                                                 level_3_filters])]

        # --------------------------------------------------------------
        # Define generative pass time connectors and combiners
        # --------------------------------------------------------------
        self.level_4_to_level_3_connector = tfl.Conv2D(filters=level_3_filters,
                                                       kernel_size=(1, 1),
                                                       strides=1,
                                                       padding="valid",
                                                       use_bias=True,
                                                       name="level_4_to_level_3_connector")

        self.level_4_to_level_2_connector = SignalConv2D(filters=level_2_filters,
                                                         kernel=(5, 5),
                                                         strides_up=4,
                                                         corr=False,
                                                         padding="reflect",
                                                         use_bias=True,
                                                         name="level_4_to_level_2_connector")

        self.level_4_to_level_1_connector = SignalConv2D(filters=level_1_filters,
                                                         kernel=(5, 5),
                                                         strides_up=4,
                                                         corr=False,
                                                         padding="reflect",
                                                         use_bias=True,
                                                         name="level_4_to_level_1_connector")

        self.level_3_to_level_2_connector = SignalConv2D(filters=level_2_filters,
                                                         kernel=(5, 5),
                                                         strides_up=4,
                                                         corr=False,
                                                         padding="reflect",
                                                         use_bias=True,
                                                         name="level_3_to_level_2_connector")

        self.level_3_to_level_1_connector = SignalConv2D(filters=level_1_filters,
                                                         kernel=(5, 5),
                                                         strides_up=4,
                                                         corr=False,
                                                         padding="reflect",
                                                         use_bias=True,
                                                         name="level_3_to_level_1_connector")

        self.level_2_to_level_1_connector = tfl.Conv2D(filters=level_1_filters,
                                                       kernel_size=(1, 1),
                                                       strides=1,
                                                       padding="valid",
                                                       use_bias=True,
                                                       name="level_2_to_level_1_connector")

        self.generative_combiners = [tfl.Conv2D(filters=filters,
                                                kernel_size=(1, 1),
                                                strides=1,
                                                padding="valid",
                                                use_bias=True,
                                                name=f"level_{i + 1}_generative_combiner")
                                     for i, filters in enumerate([level_1_filters,
                                                                  level_2_filters,
                                                                  level_3_filters,
                                                                  level_4_filters])]

        # --------------------------------------------------------------
        # Define posterior statistic combiners
        # --------------------------------------------------------------

        self.posterior_loc_combiners = [tfl.Conv2D(filters=filters,
                                                   kernel_size=(1, 1),
                                                   strides=1,
                                                   padding="valid",
                                                   use_bias=True,
                                                   name=f"level_{i + 1}_posterior_loc_combiner")
                                        for i, filters in enumerate([level_1_filters,
                                                                     level_2_filters,
                                                                     level_3_filters,
                                                                     level_4_filters])]

        self.posterior_log_scale_combiners = [tfl.Conv2D(filters=filters,
                                                         kernel_size=(1, 1),
                                                         strides=1,
                                                         padding="valid",
                                                         use_bias=True,
                                                         name=f"level_{i + 1}_posterior_log_scale_combiner")
                                              for i, filters in enumerate([level_1_filters,
                                                                           level_2_filters,
                                                                           level_3_filters,
                                                                           level_4_filters])]

    def kl_divergence(self):
        return [tf.reduce_mean(tf.reduce_sum(tfd.kl_divergence(posterior, prior), axis=[1, 2, 3]))
                for posterior, prior in [(self.posterior_1, self.prior_1),
                                         (self.posterior_2, self.prior_2),
                                         (self.posterior_3, self.prior_3),
                                         (self.posterior_4, self.prior_4)]]

    def combine_tensors(self, *args, combiner=None):
        if combiner is None:
            raise ValueError("Combiner must be specified!")

        # Assume that every element of args is going to be a 4-D tensor
        # with conformal batch size, height and width
        tensor = tf.concat(args, axis=-1)
        tensor = tf.nn.elu(tensor)

        return combiner(tensor)

    def call(self, inputs, sampling_fn=None):
        batch_size, height, width, _ = inputs.shape
        # =====================================================================
        # Inference Pass
        # =====================================================================
        tensor = inputs

        # --------------------------------------------------------------
        # Get level 1 statistics and features to pass on
        # --------------------------------------------------------------
        post_loc_1, post_log_scale_1, infer_features_1, residual_features_1 = self.analysis_transform(tensor)
        residual_features_1 = self.inputs_to_level_1_connector(residual_features_1)

        # --------------------------------------------------------------
        # Get level 2 statistics and features
        # --------------------------------------------------------------
        tensor = self.combine_tensors(residual_features_1,
                                      infer_features_1,
                                      combiner=self.inference_combiners[0])

        post_loc_2, post_log_scale_2, infer_features_2 = self.extended_analysis_transform(tensor)

        # --------------------------------------------------------------
        # Get level 3 statistics and features
        # --------------------------------------------------------------
        tensor = self.combine_tensors(self.inputs_to_level_2_connector(residual_features_1),
                                      self.level_1_to_level_2_connector(infer_features_1),
                                      infer_features_2,
                                      combiner=self.inference_combiners[1])

        post_loc_3, post_log_scale_3, infer_features_3 = self.hyper_analysis_transform(tensor)

        # --------------------------------------------------------------
        # Get level 4 statistics
        # --------------------------------------------------------------
        tensor = self.combine_tensors(self.inputs_to_level_3_connector(residual_features_1),
                                      self.level_1_to_level_3_connector(infer_features_1),
                                      self.level_2_to_level_3_connector(infer_features_2),
                                      infer_features_3,
                                      combiner=self.inference_combiners[2])

        post_loc_4, post_log_scale_4 = self.extended_hyper_analysis_transform(tensor)

        # =====================================================================
        # Generative Pass
        # =====================================================================
        prior_loc_4, prior_log_scale_4, gen_features_4 = self.empirical_hyper_prior(batch_size,
                                                                                    height // 64,
                                                                                    width // 64)
        prior_scale_4 = tf.nn.softplus(prior_log_scale_4)

        self.prior_4 = tfd.Normal(loc=prior_loc_4,
                                  scale=prior_scale_4)

        post_loc_4 = self.combine_tensors(prior_loc_4,
                                          post_loc_4,
                                          combiner=self.posterior_loc_combiners[3])

        post_log_scale_4 = self.combine_tensors(prior_log_scale_4,
                                                post_log_scale_4,
                                                combiner=self.posterior_log_scale_combiners[3])
        post_scale_4 = tf.nn.softplus(post_log_scale_4)

        self.posterior_4 = tfd.Normal(loc=post_loc_4,
                                      scale=post_scale_4)

        # Generate sample from posterior
        if sampling_fn is None:
            latent_code_4 = self.posterior_4.sample()

        else:
            indices_4, latent_code_4 = sampling_fn(target=self.posterior_4,
                                                   coder=self.prior_4)

        tensor = self.combine_tensors(latent_code_4,
                                      gen_features_4,
                                      combiner=self.generative_combiners[3])

        # --------------------------------------------------------------
        # Pass down to level 3
        # --------------------------------------------------------------
        prior_loc_3, prior_log_scale_3, gen_features_3 = self.extended_hyper_synthesis_transform(tensor)
        prior_scale_3 = tf.nn.softplus(prior_log_scale_3)

        self.prior_3 = tfd.Normal(loc=prior_loc_3,
                                  scale=prior_scale_3)

        post_loc_3 = self.combine_tensors(prior_loc_3,
                                          post_loc_3,
                                          combiner=self.posterior_loc_combiners[2])

        post_log_scale_3 = self.combine_tensors(prior_log_scale_3,
                                                post_log_scale_3,
                                                combiner=self.posterior_log_scale_combiners[2])

        post_scale_3 = tf.nn.softplus(post_log_scale_3)

        self.posterior_3 = tfd.Normal(loc=post_loc_3,
                                      scale=post_scale_3)

        # Generate sample from posterior
        if sampling_fn is None:
            latent_code_3 = self.posterior_3.sample()

        else:
            indices_3, latent_code_3 = sampling_fn(target=self.posterior_3,
                                                   coder=self.prior_3)

        tensor = self.combine_tensors(latent_code_3,
                                      gen_features_3,
                                      self.level_4_to_level_3_connector(gen_features_4),
                                      combiner=self.generative_combiners[2])
        # --------------------------------------------------------------
        # Pass down to level 2
        # --------------------------------------------------------------
        prior_loc_2, prior_log_scale_2, gen_features_2 = self.hyper_synthesis_transform(tensor)
        prior_scale_2 = tf.nn.softplus(prior_log_scale_2)

        self.prior_2 = tfd.Normal(loc=prior_loc_2,
                                  scale=prior_scale_2)

        post_loc_2 = self.combine_tensors(prior_loc_2,
                                          post_loc_2,
                                          combiner=self.posterior_loc_combiners[1])

        post_log_scale_2 = self.combine_tensors(prior_log_scale_2,
                                                post_log_scale_2,
                                                combiner=self.posterior_log_scale_combiners[1])

        post_scale_2 = tf.nn.softplus(post_log_scale_2)

        self.posterior_2 = tfd.Normal(loc=post_loc_2,
                                      scale=post_scale_2)

        # Generate sample from posterior
        if sampling_fn is None:
            latent_code_2 = self.posterior_2.sample()

        else:
            indices_2, latent_code_2 = sampling_fn(target=self.posterior_2,
                                                   coder=self.prior_2)

        tensor = self.combine_tensors(latent_code_2,
                                      gen_features_2,
                                      self.level_4_to_level_2_connector(gen_features_4),
                                      self.level_3_to_level_2_connector(gen_features_3),
                                      combiner=self.generative_combiners[1])

        # --------------------------------------------------------------
        # Pass down to level 1
        # --------------------------------------------------------------
        prior_loc_1, prior_log_scale_1, gen_features_1 = self.extended_synthesis_transform(tensor)
        prior_scale_1 = tf.nn.softplus(prior_log_scale_1)

        self.prior_1 = tfd.Normal(loc=prior_loc_1,
                                  scale=prior_scale_1)

        post_loc_1 = self.combine_tensors(prior_loc_1,
                                          post_loc_1,
                                          combiner=self.posterior_loc_combiners[0])

        post_log_scale_1 = self.combine_tensors(prior_log_scale_1,
                                                post_log_scale_1,
                                                combiner=self.posterior_log_scale_combiners[0])

        post_scale_1 = tf.nn.softplus(post_log_scale_1)

        self.posterior_1 = tfd.Normal(loc=post_loc_1,
                                      scale=post_scale_1)

        # Generate sample from posterior
        if sampling_fn is None:
            latent_code_1 = self.posterior_1.sample()

        else:
            indices_1, latent_code_1 = sampling_fn(target=self.posterior_1,
                                                   coder=self.prior_1)

        tensor = self.combine_tensors(latent_code_1,
                                      gen_features_1,
                                      self.level_4_to_level_1_connector(gen_features_4),
                                      self.level_3_to_level_1_connector(gen_features_3),
                                      self.level_2_to_level_1_connector(gen_features_2),
                                      combiner=self.generative_combiners[0])
        # --------------------------------------------------------------
        # Perform reconstruction
        # --------------------------------------------------------------
        tensor = self.synthesis_transform(tensor)

        if sampling_fn is None:
            return tensor
        else:
            return [indices_4, indices_3, indices_2, indices_1], tensor

    def compress(self, file_path, image, seed, sampler, block_size, max_index):
        pass

    def decompress(self, file_path, sampler):
        pass
