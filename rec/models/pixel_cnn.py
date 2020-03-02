import tensorflow as tf

from rec.models.custom_modules import ReparameterizedConv2D, ReparameterizedConv2DTranspose, CustomModuleError

tfl = tf.keras.layers


class PixelCNNResidualBlock(tfl.Layer):

    def __init__(self, filters, residual_filter_factor=2, name="pixel_cnn_residual_block", **kwargs):
        super().__init__(name=name,
                         **kwargs)

        self.filters = filters

        # Factor by which we increase the number of filters in the residual part
        self.residual_filter_factor = residual_filter_factor

        self.convolutions = []

    def build(self, input_shape):

        if len(input_shape) != 4:
            raise CustomModuleError(f"Input must be a rank 4 tensor! (Got shape {input_shape})")

        input_filters = input_shape[3]

        if input_filters != self.filters * self.residual_filter_factor:
            raise CustomModuleError(f"Number of input filters ({input_filters}) must be {self.residual_filter_factor}"
                                    f" times the number of filters set for the residual block ({self.filters})!")

        self.convolutions = [
            ReparameterizedConv2D(filters=self.filters,
                                  kernel_size=(1, 1),
                                  padding="same",
                                  strides=(1, 1),
                                  mask="b"),
            ReparameterizedConv2D(filters=self.filters,
                                  kernel_size=(3, 3),
                                  padding="same",
                                  strides=(1, 1),
                                  mask="b"),
            ReparameterizedConv2D(filters=self.filters * self.residual_filter_factor,
                                  kernel_size=(1, 1),
                                  padding="same",
                                  strides=(1, 1),
                                  mask="b"),
        ]

        super().build(input_shape=input_shape)

    def call(self, tensor, residual_factor=0.1):

        input = tensor

        for convolution in self.convolutions:
            tensor = tf.nn.elu(tensor)
            tensor = convolution(tensor)

        return input + residual_factor * tensor


class PixelCNN(tfl.Layer):

    def __init__(self, filters, num_residual_blocks, name="pixel_cnn", **kwargs):
        super().__init__(name=name,
                         **kwargs)

        self.filters = filters
        self.num_residual_blocks = num_residual_blocks

        # ---------------------------------------------------------------------
        # Declare layers
        # ---------------------------------------------------------------------
        self.first_conv = None
        self.residual_blocks = []
        self.last_conv = None

    def build(self, input_shape):

        if len(input_shape) != 4:
            raise CustomModuleError(f"Input must be a rank 4 tensor! (Got shape {input_shape})")

        input_filters = input_shape[3]

        self.first_conv = ReparameterizedConv2D(filters=self.filters,
                                                kernel_size=(7, 7),
                                                strides=(1, 1),
                                                padding="same",
                                                mask="a")
        self.residual_blocks = [PixelCNNResidualBlock(filters=self.filters // 2,
                                                      residual_filter_factor=2,
                                                      name=f"res_block_{res_idx}")
                                for res_idx in range(self.num_residual_blocks)]

        self.last_conv = ReparameterizedConv2D(filters=input_filters,
                                               kernel_size=(1, 1),
                                               padding="same",
                                               strides=(1, 1))

    def call(self, image):
        pass
