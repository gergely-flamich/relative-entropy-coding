import abc
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class SampleGenerator(abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abc.abstractmethod
    def get_ratios(self, target: tfd.Distribution, coder: tfd.Distribution, seed: tf.int64) -> tf.Tensor:
        """
            Fills the buffer with samples from the coder distribution.

            :param target:
            :param coder:
            :param seed:
            :return buffer of log likelihood ratios:
        """

    @abc.abstractmethod
    def get_index(self, i: tf.int64) -> tf.Tensor:
        """
            Get the sample at index i from the buffer.

            :param i:
            :return sample at index i:
        """

    @abc.abstractmethod
    def generate_index(self, i: tf.int64, coder: tfd.Distribution, seed: tf.int64) -> tf.Tensor:
        """
            Generate the buffer and then retrieve the sample at index i. Used for decoding.

            :param i:
            :param coder:
            :param seed:
            :return sample at index i:
        """


class NaiveSampleGenerator(SampleGenerator):
    def __init__(self, sample_buffer_size, **kwargs):
        super().__init__(**kwargs)
        self.sample_buffer_size = sample_buffer_size
        self.samples = None

    def get_ratios(self, target: tfd.Distribution, coder: tfd.Distribution, seed: tf.int64) -> tf.Tensor:
        tf.random.set_seed(seed)
        self.samples = coder.sample((self.sample_buffer_size,), seed=seed)
        n_axes = len(self.samples.shape)
        return tf.reduce_sum(target.log_prob(self.samples) - coder.log_prob(self.samples), axis=range(1, n_axes))

    def get_index(self, i: tf.int64) -> tf.Tensor:
        return self.samples[i]

    def generate_index(self, i: tf.int64, coder: tfd.Distribution, seed: tf.int64) -> tf.Tensor:
        tf.random.set_seed(seed)
        self.samples = coder.sample((self.sample_buffer_size,), seed=seed)
        return self.samples[i]


class PseudoSampleGenerator(SampleGenerator):
    def __init__(self, sample_buffer_size, n_true_samples=50, n_groups=50, **kwargs):
        super().__init__(**kwargs)
        self.sample_buffer_size = sample_buffer_size
        self.n_true_samples = n_true_samples
        self.n_groups = n_groups
        self.samples = None,
        self.group_assignments = None,
        self.sample_assignments = None

    def get_ratios(self, target: tfd.Distribution, coder: tfd.Distribution, seed: tf.int64) -> tf.Tensor:
        tf.random.set_seed(seed)
        self.samples = coder.sample((self.n_true_samples,), seed=seed)
        sample_ratios = target.log_prob(self.samples) - coder.log_prob(self.samples)
        flat_sample_ratios = tf.reshape(sample_ratios, (self.n_true_samples, -1))
        self.group_assignments = tf.random.uniform(coder.loc.shape, 0, self.n_groups, dtype=tf.int32, seed=seed)
        flat_group_assignments = tf.reshape(self.group_assignments, (-1,))
        mask = tf.one_hot(flat_group_assignments, self.n_groups, axis=-1)
        group_ratios = tf.matmul(flat_sample_ratios, mask)
        self.sample_assignments = tf.random.uniform((self.n_groups, self.sample_buffer_size, 1),
                                                    0,
                                                    self.n_true_samples,
                                                    dtype=tf.int32,
                                                    seed=seed)
        return tf.reduce_sum(tf.gather_nd(tf.transpose(group_ratios), self.sample_assignments, batch_dims=1), axis=0)

    def get_index(self, i: tf.int64) -> tf.Tensor:
        n_dims = len(self.samples.shape) - 1
        group_indices = self.sample_assignments[:, i, 0]
        flat_sample_indices = tf.gather_nd(group_indices, tf.reshape(self.group_assignments, (-1, 1)))
        sample = tf.gather_nd(tf.transpose(self.samples, list(range(1, n_dims + 1)) + [0]),
                              tf.reshape(flat_sample_indices, self.samples.shape[1:] + [1]),
                              batch_dims=n_dims)
        return sample

    def generate_index(self, i: tf.int64, coder: tfd.Distribution, seed: tf.int64) -> tf.Tensor:
        tf.random.set_seed(seed)
        self.samples = coder.sample((self.n_true_samples,), seed=seed)
        dims = coder.loc.shape
        n_dims = len(dims)
        self.group_assignments = tf.random.uniform(coder.loc.shape, 0, self.n_groups, dtype=tf.int32, seed=seed)
        self.sample_assignments = tf.random.uniform((self.n_groups, self.sample_buffer_size, 1),
                                                    0,
                                                    self.n_true_samples,
                                                    dtype=tf.int32,
                                                    seed=seed)
        group_indices = self.sample_assignments[:, i, 0]
        flat_sample_indices = tf.gather_nd(group_indices, tf.reshape(self.group_assignments, (-1, 1)))
        sample = tf.gather_nd(tf.transpose(self.samples, list(range(1, n_dims + 1)) + [0]),
                              tf.reshape(flat_sample_indices, self.samples.shape[1:] + [1]),
                              batch_dims=n_dims)
        return sample

















