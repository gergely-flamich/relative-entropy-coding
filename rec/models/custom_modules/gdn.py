import tensorflow as tf

tfl = tf.keras.layers


__all__ = [
    "GDN"
]


@tf.custom_gradient
def lower_bound(var, bound):

    def grad(grad_maximum):
        pass_through_if = tf.logical_or(var >= bound, grad_maximum < 0.)

        return [tf.cast(pass_through_if, grad_maximum.dtype) * grad_maximum,
                None]

    return tf.maximum(var, bound), grad


# TODO: Proper citation in docstring
class GDN(tfl.Layer):
    """
    Implements the GDN layer from Balle's papers
    """

    def __init__(self,
                 inverse: bool,
                 gamma_init: float = 0.1,
                 beta_minimum=1e-6,
                 gamma_minimum=0.,
                 reparam_offset=2 ** -18,
                 name="gdn_layer",
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self._inverse = inverse
        self.gamma_init = gamma_init
        self.beta_minimum = beta_minimum
        self.gamma_minimum = gamma_minimum
        self.reparam_offset = reparam_offset

        self.pedestal = tf.constant(self.reparam_offset ** 2, dtype=self.dtype)
        self.beta_bound = tf.constant((self.beta_minimum + self.reparam_offset ** 2) ** 0.5,
                                      dtype=self.dtype)
        self.gamma_bound = tf.constant((self.gamma_minimum + self.reparam_offset ** 2) ** 0.5,
                                       dtype=self.dtype)

        self.input_spec = tfl.InputSpec(min_ndim=2)
        self._input_rank = -1

    @property
    def inverse(self):
        return self._inverse

    @property
    def beta(self):
        return tf.square(lower_bound(self._beta, self.beta_minimum)) - self.pedestal

    @property
    def gamma(self):
        return tf.square(lower_bound(self._gamma, self.gamma_minimum)) - self.pedestal

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)

        # We assume that the last dimension is the channels' dimension for simplicity
        num_channels = input_shape.as_list()[-1]

        self._input_rank = input_shape.ndims

        if self._input_rank != 4:
            raise ValueError(f"Input rank must be 4, but had shape {input_shape}")

        self.input_spec = tfl.InputSpec(ndim=input_shape.ndims,
                                        axes={-1: num_channels})

        beta_init = tf.ones((num_channels,), dtype=self.dtype)
        beta_init = tf.sqrt(tf.maximum(beta_init + self.pedestal, self.pedestal))

        gamma_init = self.gamma_init * tf.eye(num_channels, dtype=self.dtype)
        gamma_init = tf.sqrt(tf.maximum(gamma_init + self.pedestal, self.pedestal))

        self._beta = tf.Variable(beta_init,
                                 name="beta_reparam")
        self._gamma = tf.Variable(gamma_init,
                                  name="gamma_reparam")

        super().build(input_shape)

    def call(self, tensor):
        tensor = tf.convert_to_tensor(tensor)
        tensor = tf.cast(tensor, self.dtype)

        gamma = self.gamma
        shape = gamma.shape.as_list()

        gamma = tf.reshape(gamma, [1, 1] + shape)
        norm_pool = tf.nn.convolution(tf.square(tensor),
                                      gamma,
                                      strides=(1, 1),
                                      padding="VALID",
                                      data_format="NHWC",
                                      name="gdn_conv")
        norm_pool = tf.nn.bias_add(norm_pool, self.beta)

        if self.inverse:
            norm_pool = tf.sqrt(norm_pool)

        else:
            norm_pool = tf.math.rsqrt(norm_pool)

        outputs = tensor * norm_pool

        return outputs
