import unittest
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from rec.coding.samplers import RejectionSampler


class TestRejectionSampler(unittest.TestCase):
    def test_logprob(self):
        sampler = RejectionSampler(sample_buffer_size=10000, r_buffer_size=10000)
        t = tfp.distributions.Normal(loc=tf.constant([2.]), scale=tf.constant([0.1]))
        p = tfp.distributions.Normal(loc=tf.constant([0.]), scale=tf.constant([1.]))

        logprobs = []
        for _ in range(5):
            sample, index = sampler.coded_sample(t, p, seed=42069)
            logprobs.append(t.log_prob(sample))


if __name__ == '__main__':
    unittest.main()
