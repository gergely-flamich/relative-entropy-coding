"""
Based on SignalConvND from tensorflow/compression
TODO: add proper refernce
"""

import tensorflow as tf
import numpy as np
from scipy.fftpack import rfft, irfft

tfl = tf.keras.layers

__all__ = [
    "SignalConv2D"
]


class SignalConv2D(tfl.Layer):

    def __init__(self,
                 filters,
                 kernel,
                 corr=False,
                 strides_down=1,
                 strides_up=1,
                 padding="zeros",
                 extra_pad_end=True,
                 use_bias=True,
                 dft_kernel_parametrization=True,
                 name="signal_conv2d",
                 **kwargs):
        #                  kernel_initializer=tf.initializers.variance_scaling(),
        #                  bias_initializer=tf.initializers.zeros(),

        super().__init__(name=name,
                         **kwargs)

        self.filters = int(filters)
        self._kernel_shape = kernel
        self.rank = 2
        self.corr = bool(corr)
        self.strides_down = int(strides_down)
        self.strides_up = int(strides_up)
        self.extra_pad_end = bool(extra_pad_end)
        self.use_bias = bool(use_bias)
        self.padding = str(padding).lower()
        self.dft_kernel_parametrization = bool(dft_kernel_parametrization)

        self.pad_mode = {
            "zeros": "CONSTANT",
            "reflect": "REFLECT",
        }[self.padding]

        self.input_spec = tfl.InputSpec(ndim=self.rank + 2)

        self._kernel = None
        self._bias = None

    @property
    def kernel(self):
        if not self.dft_kernel_parametrization:
            return self._kernel

        else:
            irdft_matrix = tf.py_function(calculate_irdft_matrix,
                                          inp=[self._kernel_shape],
                                          Tout=tf.float32)

            kernel = tf.linalg.matmul(irdft_matrix, self._kernel)
            kernel = tf.reshape(kernel, self._kernel_shape + (self.input_channels, self.filters))

            return kernel

    @property
    def bias(self):
        return self._bias

    def _padded_tuple(self, iterable, fill):
        return (fill,) + tuple(iterable) + (fill,)

    def build(self, input_shape):

        input_shape = tf.TensorShape(input_shape)
        channel_axis = -1
        input_channels = input_shape.as_list()[channel_axis]
        self.input_channels = input_channels

        self.input_spec = tfl.InputSpec(
            ndim=self.rank + 2, axes={channel_axis: input_channels}
        )

        kernel_shape = self._kernel_shape + (input_channels, self.filters)
        output_channels = self.filters

        # ---------------------------------------------------------------------
        # Initialize the kernel
        # ---------------------------------------------------------------------
        kernel_initializer = tf.initializers.VarianceScaling()

        if self._kernel_shape == (1, 1) or not self.dft_kernel_parametrization:
            tf.print(f"Building kernel without DCT parameterization! Shape: {kernel_shape}")
            kernel_init = kernel_initializer(kernel_shape)

        else:
            size = kernel_shape[0] * kernel_shape[1]

            # DCT parametrization
            rdft_shape = (size, kernel_shape[2] * kernel_shape[3])

            kernel_init = kernel_initializer(kernel_shape)
            kernel_init = tf.reshape(kernel_init, (-1, rdft_shape[-1]))

            irdft_matrix = tf.py_function(calculate_irdft_matrix,
                                          inp=[kernel_shape[:-2]],
                                          Tout=tf.float32)

            kernel_init = tf.linalg.matmul(irdft_matrix, kernel_init, transpose_a=True)

        self._kernel = tf.Variable(kernel_init, name="signal_conv_kernel")

        if self.use_bias:
            self._bias = tf.Variable(tf.zeros((output_channels,)), name="signal_conv_bias")

        super().build(input_shape)

    def corr_down_explicit(self, inputs, kernel, padding):
        strides = self._padded_tuple([self.strides_down, self.strides_down], 1)
        padding = self._padded_tuple(padding, (0, 0))

        return tf.nn.conv2d(inputs,
                               kernel,
                               strides=strides,
                               padding=padding,
                               data_format="NHWC")

    # Needed because of reflection padding
    def corr_down_valid(self, inputs, kernel):

        outputs = tf.nn.convolution(inputs,
                                    kernel,
                                    strides=[self.strides_down, self.strides_down],
                                    padding="VALID",
                                    data_format="NHWC")

        return outputs

    def conv_up_explicit(self, inputs, kernel, prepadding):
        # Pretend the the input filters are the output filters and vice versa
        kernel = tf.transpose(kernel, [0, 1, 3, 2])
        input_shape = inputs.shape
        padding = 4 * [(0, 0)]
        output_shape = [input_shape[0], None, None, self.filters]
        spatial_axes = range(1, 3)

        if self.extra_pad_end:
            get_length = lambda l, s, k, p: l * s + ((k - 1) - p)
        else:
            get_length = lambda l, s, k, p: l * s + ((k - 1) - (s - 1) - p)

        for i, a in enumerate(spatial_axes):
            padding[a] = (
                prepadding[i][0] * self.strides_up + self._kernel_shape[i] // 2,
                prepadding[i][1] * self.strides_up + (self._kernel_shape[i] - 1) // 2
            )

            output_shape[a] = get_length(input_shape[a],
                                         self.strides_up,
                                         self._kernel_shape[i],
                                         sum(padding[a]))

        strides = self._padded_tuple([self.strides_up, self.strides_up], 1)

        outputs = tf.compat.v1.nn.conv2d_backprop_input(output_shape,
                                                        kernel,
                                                        inputs,
                                                        strides=strides,
                                                        padding=padding,
                                                        data_format="NHWC")

        if self.strides_down > 1:
            slices = tuple(slice(None, None, self.strides_down, self.strides_down))
            slices = self._padded_tuple(slices, slice(None))
            outputs = outputs[slices]

        return outputs

    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        outputs = inputs

        kernel = self.kernel
        corr = self.corr

        # Convolution with no UPsampling
        if not corr and self.strides_up == 1:
            corr = True
            slices = self.rank * (slice(None, None, -1),) + 2 * (slice(None),)
            kernel = kernel[slices]

        # Convolution with UPsampling
        elif corr and self.strides_up != 1:
            corr = False
            slices = self.rank * (slice(None, None, -1),) + 2 * (slice(None),)
            kernel = kernel[slices]

        # Padding
        padding = same_padding_for_kernel(self._kernel_shape,
                                          corr=corr,
                                          strides_up=[self.strides_up, self.strides_up])

        if self.padding == "zeros":
            prepadding = self.rank * ((0, 0),)

        else:
            outputs = tf.pad(outputs, self._padded_tuple(padding, (0, 0)), self.pad_mode)
            prepadding = padding
            padding = self.rank * ((0, 0),)

        # ---------------------------------------------------------------------
        # Perform the convolution
        # ---------------------------------------------------------------------
        if corr and self.strides_up == 1 and not all(p[0] == p[1] == 0 for p in padding):
            outputs = self.corr_down_explicit(outputs, kernel, padding)

        elif corr and self.strides_up == 1 and all(p[0] == p[1] == 0 for p in padding):
            outputs = self.corr_down_valid(outputs, kernel)

        elif not corr:
            # Explicit up-convolution
            outputs = self.conv_up_explicit(outputs, kernel, prepadding)

        else:
            raise NotImplementedError

        # ---------------------------------------------------------------------
        # Add bias
        # ---------------------------------------------------------------------
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        return outputs


def calculate_irdft_matrix(shape, dtype=tf.float32):
  """
  Taken verbatim from tensorflow_compression.ops.spectral_ops
  """
  shape = tuple(int(s) for s in shape)
  dtype = tf.as_dtype(dtype)
  size = np.prod(shape)
  rank = len(shape)
  matrix = np.identity(size, dtype=np.float64).reshape((size,) + shape)
  for axis in range(rank):
    matrix = rfft(matrix, axis=axis + 1)
    slices = (rank + 1) * [slice(None)]
    if shape[axis] % 2 == 1:
      slices[axis + 1] = slice(1, None)
    else:
      slices[axis + 1] = slice(1, -1)
    matrix[tuple(slices)] *= np.sqrt(2)
  matrix /= np.sqrt(size)
  matrix = np.reshape(matrix, (size, size))
  return tf.constant(
      matrix, dtype=dtype, name="irdft_" + "x".join([str(s) for s in shape]))


def same_padding_for_kernel(shape, corr, strides_up=None):
  """
  Taken verbatim from tensorflow_compression.ops.padding_ops
  """
  rank = len(shape)
  if strides_up is None:
    strides_up = rank * (1,)

  if corr:
    padding = [(s // 2, (s - 1) // 2) for s in shape]
  else:
    padding = [((s - 1) // 2, s // 2) for s in shape]

  padding = [((padding[i][0] - 1) // strides_up[i] + 1,
              (padding[i][1] - 1) // strides_up[i] + 1) for i in range(rank)]
  return padding


class IdentityInitializer(object):
  """Initialize to the identity kernel with the given shape.

  This creates an n-D kernel suitable for `SignalConv*` with the requested
  support that produces an output identical to its input (except possibly at the
  signal boundaries).

  Note: The identity initializer in `tf.initializers` is only suitable for
  matrices, not for n-D convolution kernels (i.e., no spatial support).
  """

  def __init__(self, gain=1):
    self.gain = float(gain)

  def __call__(self, shape, dtype=None, partition_info=None):
    del partition_info  # unused
    assert len(shape) > 2, shape

    support = tuple(shape[:-2]) + (1, 1)
    indices = [[s // 2 for s in support]]
    updates = tf.constant([self.gain], dtype=dtype)
    kernel = tf.scatter_nd(indices, updates, support)

    assert shape[-2] == shape[-1], shape
    if shape[-1] != 1:
      kernel = kernel * tf.eye(shape[-1], dtype=tf.float32)

    return kernel