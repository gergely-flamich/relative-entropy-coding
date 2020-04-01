import unittest
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from rec.coding.sample_generator import NaiveSampleGenerator, PseudoSampleGenerator


class TestSampleGenerator(unittest.TestCase):

    def test_naive_sampling(self):
        t = tfp.distributions.Normal(loc=tf.constant(np.repeat(0.1, 10), dtype=tf.float32),
                                     scale=tf.constant(np.repeat(0.1, 10), dtype=tf.float32))
        p = tfp.distributions.Normal(loc=tf.constant(np.repeat(0., 10), dtype=tf.float32),
                                     scale=tf.constant(np.repeat(1., 10), dtype=tf.float32))

        naive_sg = NaiveSampleGenerator(100000)
        log_ratios = naive_sg.get_ratios(t, p, 420)
        i = 5
        sample_i = naive_sg.get_index(i)
        self.assertAlmostEqual(log_ratios[i], tf.reduce_sum(t.log_prob(sample_i) - p.log_prob(sample_i)))
        sample_i = naive_sg.generate_index(i, p, 420)
        self.assertAlmostEqual(log_ratios[i], tf.reduce_sum(t.log_prob(sample_i) - p.log_prob(sample_i)))

    def test_pseudo_sampling(self):
        t = tfp.distributions.Normal(loc=tf.constant(np.repeat(0.1, 10), dtype=tf.float32),
                                     scale=tf.constant(np.repeat(0.1, 10), dtype=tf.float32))
        p = tfp.distributions.Normal(loc=tf.constant(np.repeat(0., 10), dtype=tf.float32),
                                     scale=tf.constant(np.repeat(1., 10), dtype=tf.float32))

        pseudo_sg = PseudoSampleGenerator(100000)
        log_ratios = pseudo_sg.get_ratios(t, p, 420)
        i = 5
        sample_i = pseudo_sg.get_index(i)
        self.assertAlmostEqual(log_ratios[i], tf.reduce_sum(t.log_prob(sample_i) - p.log_prob(sample_i)))
        sample_i = pseudo_sg.generate_index(i, p, 420)
        self.assertAlmostEqual(log_ratios[i], tf.reduce_sum(t.log_prob(sample_i) - p.log_prob(sample_i)))


if __name__ == '__main__':
    unittest.main()