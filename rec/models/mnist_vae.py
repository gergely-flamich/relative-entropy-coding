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

    def __init__(self, latents, hidden_size, name="mnist_encoder", **kwargs):
        super(MNISTEncoder, self).__init__(name=name, **kwargs)

        self.latents = latents
        self.hidden_size = hidden_size

        self.layers = []
        self.loc_head = None
        self.log_scale_head = None

    def build(self, input_size):
        self.layers = [
            tfl.Flatten(),
            tfl.Dense(units=self.hidden_size),
            tf.nn.tanh,
            tfl.Dense(units=self.hidden_size),
            tf.nn.tanh
        ]

        self.loc_head = tfl.Dense(units=self.latents)
        self.log_scale_head = tfl.Dense(units=self.latents)

    def call(self, tensor):
        for layer in self.layers:
            tensor = layer(tensor)

        loc = self.loc_head(tensor)

        # add on small epsilon so that the variance is never 0
        scale = tf.nn.softplus(self.log_scale_head(tensor)) + 1e-5

        return loc, scale


class MNISTDecoder(tfl.Layer):

    def __init__(self, hidden_size, name="mnist_decoder", **kwargs):
        super(MNISTDecoder, self).__init__(name=name, **kwargs)

        self.hidden_size = hidden_size

        self.layers = []

    def build(self, input_size):
        self.layers = [
            tfl.Dense(units=self.hidden_size),
            tf.nn.tanh,
            tfl.Dense(units=self.hidden_size),
            tf.nn.tanh,
            tfl.Dense(units=28 * 28),
            tf.nn.sigmoid,
            tfl.Reshape((28, 28, 1))
        ]

    def call(self, tensor):
        for layer in self.layers:
            tensor = layer(tensor)

        return tensor


class MNISTVAE(tf.keras.Model):

    def __init__(self, prior, hidden_size=300, name="mnist_vae", **kwargs):
        super(MNISTVAE, self).__init__(name=name, **kwargs)

        self.hidden_size = hidden_size

        self.latents = prior.batch_shape[0]
        self.prior = prior
        self.posterior = None
        self.likelihood = None
        self.kl_divergence = 0.

        self.encoder = MNISTEncoder(latents=self.latents, hidden_size=self.hidden_size)
        self.decoder = MNISTDecoder(hidden_size=self.hidden_size)

    def call(self, tensor, training=False):
        loc, scale = self.encoder(tensor)

        self.posterior = tfd.Normal(loc=loc, scale=scale)

        code = self.posterior.sample()

        if training:
            self.prior_log_liks = self.prior.log_prob(code)
            self.post_log_liks = self.posterior.log_prob(code)

            self.kl_divergence = self.posterior.log_prob(code) - self.prior.log_prob(code)

        reconstruction = self.decoder(code)

        # if we massively mis-predict the mean of a single pixel, its log-likelihood might become very large,
        # hence we clip the means to a reasonable range first
        clipped_reconstruction = tf.clip_by_value(reconstruction, 1e-10, 1 - 1e-10)

        self.likelihood = tfd.Bernoulli(probs=clipped_reconstruction, dtype=tf.float32)

        return reconstruction


class MNISTVampVAE(tf.keras.Model):

    def __init__(self, latents, num_vamp_components=500, hidden_size=300, name="mnist_vamp_vae", **kwargs):
        super(MNISTVampVAE, self).__init__(name=name, **kwargs)

        self.hidden_size = hidden_size
        self.latents = latents
        self.num_vamp_components = num_vamp_components

        self.encoder = MNISTEncoder(latents=self.latents, hidden_size=self.hidden_size)
        self.decoder = MNISTDecoder(hidden_size=self.hidden_size)

        self.likelihood = None
        self.posterior = None
        self.kl_divergence = 0.

        # Initialize inducing points to random white noise images
        self.inducing_points = tf.Variable(tf.random.uniform(shape=(num_vamp_components, 28, 28, 1),
                                                             minval=0.,
                                                             maxval=1.))

    def call(self, tensor, training=False):

        loc, scale = self.encoder(tensor)

        self.posterior = tfd.Normal(loc=loc, scale=scale)

        code = self.posterior.sample()

        if training:

            prior_components_loc, prior_components_scale = self.encoder(self.inducing_points)

            # tfd.MixtureSameFamily expects the component dimension to be the last one, so we permute it to be the last.
            prior_components = tfd.Normal(loc=tf.transpose(prior_components_loc),
                                          scale=tf.transpose(prior_components_scale))

            self.prior = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=[1. / self.num_vamp_components] * self.num_vamp_components),
                components_distribution=prior_components
            )

            # All mixture weights are set to be equal
            self.prior_log_liks = self.prior.log_prob(code)
            self.post_log_liks = self.posterior.log_prob(code)

            # We get the empirical KL for each dimension and then sum it along the dimensions
            self.kl_divergence = self.post_log_liks - self.prior_log_liks

        reconstruction = self.decoder(code)

        # if we massively mis-predict the mean of a single pixel, its log-likelihood might become very large,
        # hence we clip the means to a reasonable range first
        clipped_reconstruction = tf.clip_by_value(reconstruction, 1e-6, 1 - 1e-6)
        self.likelihood = tfd.Bernoulli(probs=clipped_reconstruction, dtype=tf.float32)

        return reconstruction
