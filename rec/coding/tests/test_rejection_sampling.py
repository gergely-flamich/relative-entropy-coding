import unittest
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from rec.coding.rejection_sampling import get_t_p_mass, get_r_pstar


class TestRejectionSampling(unittest.TestCase):
    # Slow version for testing purposes
    def get_r_pstar_baseline(self, log_ratios, t_mass, p_mass, r_buffer_size):
        ratios_np = tf.exp(log_ratios).numpy()
        t_cummass_np = tf.exp(tf.math.cumulative_logsumexp(t_mass)).numpy()
        p_cummass_np = tf.exp(tf.math.cumulative_logsumexp(p_mass)).numpy()
        p_zero = float(1. - np.exp(tf.reduce_logsumexp(p_mass)))
        r_buffer = np.zeros((r_buffer_size,))
        pstar_buffer = np.zeros((r_buffer_size,))
        r = 0.
        pstar = 0.
        r_ind = 0
        for i in range(r_buffer_size):
            r += 1. - pstar
            r_buffer[i] = r
            while ratios_np[r_ind] < r:
                r_ind += 1
            p_cum = p_zero + (p_cummass_np[r_ind - 1] if r_ind > 0 else 0.)
            t_cum = t_cummass_np[r_ind - 1] if r_ind > 0 else 0.
            pstar = (1. - p_cum) * r + t_cum
            pstar_buffer[i] = pstar
        return r_buffer, pstar_buffer

    def test_r_pstar(self):
        t = tfp.distributions.Normal(loc=tf.constant([3.]), scale=tf.constant([0.001]))
        p = tfp.distributions.Normal(loc=tf.constant([0.]), scale=tf.constant([1.]))
        log_ratios, t_mass, p_mass = get_t_p_mass(t, p, n_samples=10, oversampling=10)
        r_buffer, pstar_buffer = get_r_pstar(log_ratios, t_mass, p_mass, r_buffer_size=10000, dtype=tf.float64)
        r_buffer_baseline, pstar_buffer_baseline = self.get_r_pstar_baseline(log_ratios,
                                                                             t_mass,
                                                                             p_mass,
                                                                             r_buffer_size=10000)
        np.testing.assert_almost_equal(r_buffer.numpy(), r_buffer_baseline, rtol=1e-5)
        np.testing.assert_almost_equal(pstar_buffer.numpy(), pstar_buffer_baseline, rtol=1e-5)

        log_ratios, t_mass, p_mass = get_t_p_mass(t, p, n_samples=2, oversampling=10)
        r_buffer, pstar_buffer = get_r_pstar(log_ratios, t_mass, p_mass, r_buffer_size=100000, dtype=tf.float64)
        r_buffer_baseline, pstar_buffer_baseline = self.get_r_pstar_baseline(log_ratios,
                                                                             t_mass,
                                                                             p_mass,
                                                                             r_buffer_size=100000)
        np.testing.assert_almost_equal(r_buffer.numpy(), r_buffer_baseline, rtol=1e-5)
        np.testing.assert_almost_equal(pstar_buffer.numpy(), pstar_buffer_baseline, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
