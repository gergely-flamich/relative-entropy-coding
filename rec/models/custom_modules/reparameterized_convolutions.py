import tensorflow as tf

# Imports from the Keras backend so we don't have to work too hard
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import activations

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.eager import context
from tensorflow.python.keras import backend

import numpy as np


class CustomModuleError(Exception):
    pass


# Taken exactly from
# https://github.com/hilloc-submission/hilloc/blob/b89e9c983e3764798e7c6f81f5cfc1d11b349d96/experiments/rvae/tf_utils/layers.py#L122
def get_linear_ar_mask(n_in, n_out, zerodiagonal=False):
    assert n_in % n_out == 0 or n_out % n_in == 0, "%d - %d" % (n_in, n_out)

    mask = np.ones([n_in, n_out], dtype=np.float32)
    if n_out >= n_in:
        k = n_out // n_in
        for i in range(n_in):
            mask[i + 1:, i * k:(i + 1) * k] = 0
            if zerodiagonal:
                mask[i:i + 1, i * k:(i + 1) * k] = 0
    else:
        k = n_in // n_out
        for i in range(n_out):
            mask[(i + 1) * k:, i:i + 1] = 0
            if zerodiagonal:
                mask[i * k:(i + 1) * k:, i:i + 1] = 0
    return mask


def get_conv_ar_mask(h, w, n_in, n_out, zerodiagonal=False):
    l = (h - 1) // 2
    m = (w - 1) // 2
    mask = np.ones([h, w, n_in, n_out], dtype=np.float32)
    mask[:l, :, :, :] = 0
    mask[l, :m, :, :] = 0
    mask[l, m, :, :] = get_linear_ar_mask(n_in, n_out, zerodiagonal)
    return mask


class ReparameterizedConv(tf.keras.layers.Layer):

    def __init__(self, rank,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 mask=None,
                 **kwargs):
        super().__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(
            dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)

        self.mask_type = mask
        self._mask = None

        # Initialization flag, because the first pass has to be calculated "using batch norm"
        self._initialized = tf.Variable(False, name="initialization_flag", trainable=False)

        self.kernel_shape = None

        self.kernel_weights = None
        self.kernel_log_scale = None
        self.bias = None

    def _get_kernel(self, norm_axes, initializing=False):
        v = tf.math.l2_normalize(self.kernel_weights, axis=norm_axes)

        v = v * self.mask

        if not initializing:
            v = v * tf.exp(self.kernel_log_scale)

        return v

    def _get_mask(self, axes_hwio):
        """

        :param axes_hwio: the axes in [height, width, input filters, output filters] order
        :return:
        """

        kernel_shape = [self.kernel_weights.shape[i] for i in axes_hwio]

        # invert the axis permutation
        inv_perm = [0] * len(axes_hwio)

        for i in range(len(axes_hwio)):
            inv_perm[axes_hwio[i]] = i

        if self.mask_type is None:
            return tf.ones(self.kernel_weights.shape)

        if self.mask_type == "a":
            mask = tf.convert_to_tensor(get_conv_ar_mask(*kernel_shape, zerodiagonal=True))
        elif self.mask_type == "b":
            mask = tf.convert_to_tensor(get_conv_ar_mask(*kernel_shape, zerodiagonal=False))
        else:
            raise CustomModuleError(f"Unrecognized convolution mask: {self._mask}")

        return tf.transpose(mask, perm=inv_perm)

    @property
    def mask(self):
        if self._mask is None:
            self._mask = self._get_mask([0, 1, 2, 3])

        return self._mask

    @property
    def kernel(self):
        return self._get_kernel([0, 1, 2])

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        kernel_shape = self.kernel_size + (input_channel, self.filters)

        self.kernel_weights = self.add_weight(
            name='kernel_weights',
            shape=kernel_shape,
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05),
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        self.kernel_log_scale = self.add_weight(name='kernel_log_scale',
                                                shape=(1, 1, 1, self.filters),
                                                initializer=tf.constant_initializer(value=0.),
                                                regularizer=None,
                                                constraint=None,
                                                trainable=True,
                                                dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None

        channel_axis = self._get_channel_axis()
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_channel})

        self._build_conv_op_input_shape = input_shape
        self._build_input_channel = input_channel
        self._padding_op = self._get_padding_op()
        self._conv_op_data_format = conv_utils.convert_data_format(
            self.data_format, self.rank + 2)
        self._convolution_op = nn_ops.Convolution(
            input_shape,
            filter_shape=self.kernel.shape,
            dilation_rate=self.dilation_rate,
            strides=self.strides,
            padding=self._padding_op,
            data_format=self._conv_op_data_format)
        self.built = True

    def call(self, inputs, init_scale=0.1):
        # Check if the input_shape in call() is different from that in build().
        # If they are different, recreate the _convolution_op to avoid the stateful
        # behavior.
        call_input_shape = inputs.get_shape()
        recreate_conv_op = (
                call_input_shape[1:] != self._build_conv_op_input_shape[1:])

        if recreate_conv_op:
            self._convolution_op = nn_ops.Convolution(
                call_input_shape,
                filter_shape=self.kernel.shape,
                dilation_rate=self.dilation_rate,
                strides=self.strides,
                padding=self._padding_op,
                data_format=self._conv_op_data_format)

        # Apply causal padding to inputs for Conv1D.
        if self.padding == 'causal' and self.__class__.__name__ == 'Conv1D':
            inputs = array_ops.pad(inputs, self._compute_causal_padding())

        # ---------------------------------------------------------------------
        # Check if the convolution has been initialized. If it has not,
        # perform "batch norm" and initialize the kernel scale and the bias
        # ---------------------------------------------------------------------
        if not self._initialized:
            # Disables using the scale parameter, so we only use v / ||v||
            kernel = self._get_kernel(norm_axes=[0, 1, 2], initializing=True)
            outputs = self._convolution_op(inputs, kernel)

            # Moments are calculated along batch, width and height dimensions
            out_mean, out_var = tf.nn.moments(outputs, axes=[0, 1, 2], keepdims=True)

            scale_init = init_scale / tf.sqrt(out_var + 1e-10)

            # Batch norm
            outputs = (outputs - out_mean) * scale_init

            # Initialize the kernel scale and the bias
            self.kernel_log_scale.assign(tf.math.log(scale_init) / 3.0)

            if self.use_bias:
                self.bias.assign(tf.reshape(-out_mean * scale_init, [self.filters]))

            self._initialized.assign(True)

        else:
            outputs = self._convolution_op(inputs, self.kernel)

        # If the convolution is not initialized yet, we shouldn't add on the bias
        if self.use_bias:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                    outputs += bias
                else:
                    outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
            else:
                outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return tensor_shape.TensorShape([input_shape[0]] + new_space +
                                            [self.filters])
        else:
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return tensor_shape.TensorShape([input_shape[0], self.filters] +
                                            new_space)

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _compute_causal_padding(self):
        """Calculates padding for 'causal' option for 1-d conv layers."""
        left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
        if self.data_format == 'channels_last':
            causal_padding = [[0, 0], [left_pad, 0], [0, 0]]
        else:
            causal_padding = [[0, 0], [0, 0], [left_pad, 0]]
        return causal_padding

    def _get_channel_axis(self):
        if self.data_format == 'channels_first':
            return 1
        else:
            return -1

    def _get_input_channel(self, input_shape):
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        return int(input_shape[channel_axis])

    def _get_padding_op(self):
        if self.padding == 'causal':
            op_padding = 'valid'
        else:
            op_padding = self.padding
        if not isinstance(op_padding, (list, tuple)):
            op_padding = op_padding.upper()
        return op_padding


class ReparameterizedConv2D(ReparameterizedConv):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 mask=None,
                 **kwargs):
        super().__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            mask=mask,
            **kwargs)


class ReparameterizedConv2DTranspose(ReparameterizedConv2D):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 output_padding=None,
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 mask=None,
                 **kwargs):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs)

        self.output_padding = output_padding
        if self.output_padding is not None:
            self.output_padding = conv_utils.normalize_tuple(
                self.output_padding, 2, 'output_padding')
            for stride, out_pad in zip(self.strides, self.output_padding):
                if out_pad >= stride:
                    raise ValueError('Stride ' + str(self.strides) + ' must be '
                                                                     'greater than output padding ' +
                                     str(self.output_padding))

    @property
    def kernel(self):
        return self._get_kernel([0, 1, 3])

    @property
    def mask(self):
        if self._mask is None:
            self._mask = self._get_mask([0, 1, 3, 2])

        return self._mask

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank 4. Received input shape: ' +
                             str(input_shape))
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        kernel_shape = self.kernel_size + (self.filters, input_dim)

        self.kernel_shape = kernel_shape

        self.kernel_weights = self.add_weight(
            name='kernel_weights',
            shape=kernel_shape,
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05),
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        self.kernel_log_scale = self.add_weight(name='kernel_log_scale',
                                                shape=(1, 1, self.filters, 1),
                                                initializer=tf.constant_initializer(value=0.),
                                                regularizer=None,
                                                constraint=None,
                                                trainable=True,
                                                dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)

        self.built = True

    def call(self, inputs, init_scale=0.1):
        inputs_shape = array_ops.shape(inputs)
        batch_size = inputs_shape[0]
        if self.data_format == 'channels_first':
            h_axis, w_axis = 2, 3
        else:
            h_axis, w_axis = 1, 2

        height, width = inputs_shape[h_axis], inputs_shape[w_axis]
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        if self.output_padding is None:
            out_pad_h = out_pad_w = None
        else:
            out_pad_h, out_pad_w = self.output_padding

        # Infer the dynamic output shape:
        out_height = conv_utils.deconv_output_length(height,
                                                     kernel_h,
                                                     padding=self.padding,
                                                     output_padding=out_pad_h,
                                                     stride=stride_h,
                                                     dilation=self.dilation_rate[0])
        out_width = conv_utils.deconv_output_length(width,
                                                    kernel_w,
                                                    padding=self.padding,
                                                    output_padding=out_pad_w,
                                                    stride=stride_w,
                                                    dilation=self.dilation_rate[1])
        if self.data_format == 'channels_first':
            output_shape = (batch_size, self.filters, out_height, out_width)
        else:
            output_shape = (batch_size, out_height, out_width, self.filters)

        output_shape_tensor = array_ops.stack(output_shape)

        # ---------------------------------------------------------------------
        # Check if the convolution has been initialized. If it has not,
        # perform "batch norm" and initialize the kernel scale and the bias
        # ---------------------------------------------------------------------
        if not self._initialized:
            # Do not use the kernel log scale for the initialization round
            # Remember that the filter axis for deconvolutions is the 3rd one not the 4th
            # Note: the original IAF implementation is consciously using a bugged implementation, where
            # the norm is taken over the axes [0, 1, 2], but apparently this causes training instability
            # https://github.com/openai/iaf/issues/7
            # https://github.com/openai/iaf/blob/ad33fe4872bf6e4b4f387e709a625376bb8b0d9d/tf_utils/layers.py#L93
            kernel = self._get_kernel(norm_axes=[0, 1, 3], initializing=True)
            outputs = backend.conv2d_transpose(inputs,
                                               kernel,
                                               output_shape_tensor,
                                               strides=self.strides,
                                               padding=self.padding,
                                               data_format=self.data_format,
                                               dilation_rate=self.dilation_rate)

            out_mean, out_var = tf.nn.moments(outputs, axes=[0, 1, 2], keepdims=True)

            init_scale = init_scale / tf.sqrt(out_var + 1e-10)

            # Batch norm
            normed_outputs = (outputs - out_mean) * init_scale

            # Initialize the kernel scale and the bias
            self.kernel_log_scale.assign(tf.transpose(tf.math.log(init_scale) / 3.0, perm=[0, 1, 3, 2]))

            if self.use_bias:
                self.bias.assign(tf.reshape(-out_mean * init_scale, [self.filters]))

            self._initialized.assign(True)

            return normed_outputs

        else:
            outputs = backend.conv2d_transpose(inputs,
                                               self.kernel,
                                               output_shape_tensor,
                                               strides=self.strides,
                                               padding=self.padding,
                                               data_format=self.data_format,
                                               dilation_rate=self.dilation_rate)

        # If the convolution has not been initialized yet, we shouldn't add on the bias
        if self.use_bias:
            outputs = nn.bias_add(
                outputs,
                self.bias,
                data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        output_shape = list(input_shape)
        if self.data_format == 'channels_first':
            c_axis, h_axis, w_axis = 1, 2, 3
        else:
            c_axis, h_axis, w_axis = 3, 1, 2

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        if self.output_padding is None:
            out_pad_h = out_pad_w = None
        else:
            out_pad_h, out_pad_w = self.output_padding

        output_shape[c_axis] = self.filters
        output_shape[h_axis] = conv_utils.deconv_output_length(
            output_shape[h_axis],
            kernel_h,
            padding=self.padding,
            output_padding=out_pad_h,
            stride=stride_h,
            dilation=self.dilation_rate[0])
        output_shape[w_axis] = conv_utils.deconv_output_length(
            output_shape[w_axis],
            kernel_w,
            padding=self.padding,
            output_padding=out_pad_w,
            stride=stride_w,
            dilation=self.dilation_rate[1])
        return tensor_shape.TensorShape(output_shape)

    def get_config(self):
        config = super().get_config()
        config['output_padding'] = self.output_padding
        return config


class AutoRegressiveMultiConv2D(tf.keras.layers.Layer):

    def __init__(self,
                 convolution_filters,
                 head_filters,
                 kernel_size=(3, 3),
                 name="autoregressive_multi_convolution_2d",
                 **kwargs):
        super().__init__(name=name,
                         **kwargs)

        self.convolution_filters = convolution_filters
        self.head_filters = head_filters
        self.kernel_size = kernel_size

        self.convolutions = [ReparameterizedConv2D(filters=filters,
                                                   kernel_size=self.kernel_size,
                                                   strides=(1, 1),
                                                   padding="same",
                                                   mask="b")
                             for filters in self.convolution_filters]

        self.heads = [ReparameterizedConv2D(filters=filters,
                                            kernel_size=self.kernel_size,
                                            strides=(1, 1),
                                            padding="same",
                                            mask="a")
                      for filters in self.head_filters]

    def call(self, tensor, context):

        for i, conv in enumerate(self.convolutions):
            tensor = conv(tensor)

            if i == 0:
                tensor = tensor + context

            tensor = tf.nn.elu(tensor)

        return [head(tensor) for head in self.heads]

