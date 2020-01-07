import logging

import tensorflow as tf
tfl = tf.keras.layers

import tensorflow_probability as tfp
tfd = tfp.distributions


class MNISTEncoder(tfl.Layer):
    """
    Encoder architecture used in
    REVISITING AUXILIARY LATENT VARIABLES IN GENERATIVE MODELS, D. Lawson et al., ICLR 2019
    """

    def __init__(self, latents=50, name="mnist_encoder", **kwargs):

        super(MNISTEncoder, self).__init__(name=name, **kwargs)

        self.latents = latents

        self.layers = []
        self.loc_head = None
        self.log_scale_head = None

    def build(self, input_size):

        self.layers = [
            tfl.Flatten(),
            tfl.Dense(units=300,
                      activation="tanh"),
            tfl.Dense(units=300,
                      activation="tanh")
        ]

        self.loc_head = tfl.Dense(units=self.latents)
        self.log_scale_head = tfl.Dense(units=self.latents)

        super(MNISTEncoder, self).build(input_size)

    def call(self, tensor):

        for layer in self.layers:
            tensor = layer(tensor)

        loc = self.loc_head(tensor)
        scale = tf.nn.softplus(self.log_scale_head(tensor)) + 1e-10

        return loc, scale


class MNISTDecoder(tfl.Layer):

    def __init__(self, name="mnist_decoder", **kwargs):
        super(MNISTDecoder, self).__init__(name=name, **kwargs)

        self.layers = []

    def build(self, input_size):

        self.layers = [
            tfl.Dense(units=300,
                      activation="tanh"),
            tfl.Dense(units=300,
                      activation="tanh"),
            tfl.Dense(units=28 * 28,
                      activation="sigmoid"),
            tfl.Reshape((28, 28, 1))
        ]

        super(MNISTDecoder, self).build(input_size)

    def call(self, tensor):

        for layer in self.layers:
            tensor = layer(tensor)

        return tensor


class MNISTVAE(tf.keras.Model):

    def __init__(self, prior, name="mnist_vae", **kwargs):

        super(MNISTVAE, self).__init__(name=name, **kwargs)

        self.latents = prior.batch_shape[0]
        self.prior = prior
        self.posterior = None
        self.likelihood = None
        self.kl_divergence = 0.

        self.encoder = MNISTEncoder(latents=self.latents)
        self.decoder = MNISTDecoder()

    def call(self, tensor, training=False):

        loc, scale = self.encoder(tensor)

        self.posterior = tfd.Normal(loc=loc, scale=scale)

        code = self.posterior.sample()

        if training:
            self.kl_divergence = self.posterior.log_prob(code) - self.prior.log_prob(code)

        reconstruction = self.decoder(code)

        self.likelihood = tfd.Bernoulli(probs=reconstruction, dtype=tf.float32)

        return reconstruction
