import unittest
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from rec.coding.coder import GaussianEncoder


class TestCoder(unittest.TestCase):
    def test_rs_gaussian(self):
        pass
        # encoder = GaussianEncoder(sampler='rejection')
        # t = tfp.distributions.Normal(loc=tf.constant([5.]), scale=tf.constant([0.01]))
        # p = tfp.distributions.Normal(loc=tf.constant([0.]), scale=tf.constant([1.]))
        # print(tfp.distributions.kl_divergence(t, p))



if __name__ == '__main__':
    unittest.main()
