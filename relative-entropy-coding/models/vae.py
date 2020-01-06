import logging

import tensorflow as tf
tfl = tf.keras.layers

import tensorflow_probability as tfp
tfd = tfp.distributions


class Encoder(tfl.Layer):

    def __init__(self, hidden, name="encoder", **kwargs):

        super(Encoder, self).__init__(name=name,
                                      **kwargs)

        self.hidden = hidden

    def build(self, input_size):

        self.layers = [
            tfl.Conv2D(filters=32)
        ]

    def call(self, tensor, training=False):

        for layer in self.layers:
            tensor = layer(tensor)

