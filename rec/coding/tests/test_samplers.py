import unittest
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from rec.coding.samplers import RejectionSampler


class TestRejectionSampler(unittest.TestCase):
    def test_logprob(self):
        sampler = RejectionSampler(sample_buffer_size=10000, r_buffer_size=10000)
        t = tfp.distributions.Normal(loc=tf.constant([[[2.]]]), scale=tf.constant([[[0.01]]]))
        p = tfp.distributions.Normal(loc=tf.constant([[[0.]]]), scale=tf.constant([[[1.]]]))

        logprobs = []
        for _ in range(5):
            index, sample = sampler.coded_sample(t, p, seed=42069)
            logprobs.append(t.log_prob(sample) - p.log_prob(sample))

        self.assertGreater(np.mean(logprobs), 0.)

    def test_decode(self):
        sampler = RejectionSampler(sample_buffer_size=10000, r_buffer_size=10000)
        t = tfp.distributions.Normal(loc=tf.constant([[[2.]]]), scale=tf.constant([[[0.01]]]))
        p = tfp.distributions.Normal(loc=tf.constant([[[0.]]]), scale=tf.constant([[[1.]]]))

        index, sample = sampler.coded_sample(t, p, seed=42069)
        reconstructed_sample = sampler.decode_sample(p, index, seed=42069)

        np.testing.assert_allclose(reconstructed_sample, sample)

    def test_codelength(self):
        sampler = RejectionSampler(sample_buffer_size=10000, r_buffer_size=10000)
        t = tfp.distributions.Normal(loc=tf.constant([2.]), scale=tf.constant([0.01]))
        p = tfp.distributions.Normal(loc=tf.constant([0.]), scale=tf.constant([1.]))

        index, sample = sampler.coded_sample(t, p, seed=42069)
        sampler.update(t, p)
        sampler.update(t, p)
        self.assertAlmostEqual(tf.reduce_sum(sampler.acceptance_probabilities) + sampler.spillover_probability, 1.)
        sampler.get_codelength(index)


if __name__ == '__main__':
    unittest.main()
