import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


class SNISDistribution(tf.keras.Model):

    def __init__(self, energy_fn, prior, K=128, name="snis_distribution", **kwargs):

        super(SNISDistribution, self).__init__(name=name, **kwargs)

        self.energy_fn = energy_fn
        self.prior = prior

        self.K = K
        self.log_K = tf.math.log(tf.cast(self.K, tf.float32))

    @property
    def batch_shape(self):
        return self.prior.batch_shape

    def sample(self):

        samples = self.prior.sample(self.K)
        weights = self.energy_fn(samples)

        cat = tfd.Categorical(logits=tf.reshape(weights, [-1]))
        i = cat.sample()

        return samples[i, ...]

    def log_prob(self, x):
        return self.log_prob_lower_bound(x)

    def log_prob_lower_bound(self, x):

        # Make sure x is rank-2
        x = tf.reshape(x, [-1] + self.prior.batch_shape.as_list())

        # Get samples from pi(x) and calculate their importance weights
        samples = self.prior.sample(self.K - 1)
        weights = self.energy_fn(samples)

        # Calculate the energy and pi(x) density of the given sample x
        x_energy = self.energy_fn(x)
        x_pi_log_lik = tf.reduce_sum(self.prior.log_prob(x), axis=1, keepdims=True)

        # Calculate the normalizing constant for the lower bound
        tiled_weights = tf.transpose(tf.tile(weights, [1, x.shape[0]]))
        importance_normalizer = tf.reduce_logsumexp(tf.concat((tiled_weights, x_energy), axis=1), axis=1, keepdims=True)

        # Calculate the lower bound
        return x_energy + x_pi_log_lik + self.log_K - importance_normalizer
