import abc
from typing import Tuple, List

import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np

from rec.coding.importance_sampling import encode_gaussian_importance_sample, decode_gaussian_importance_sample
from rec.coding.rejection_sampling import gaussian_rejection_sample_small, get_r_pstar, get_t_p_mass
from rec.coding.sample_generator import NaiveSampleGenerator, PseudoSampleGenerator

tfd = tfp.distributions


class Sampler(tf.Module, abc.ABC):

    def __init__(self, name="sampler", **kwargs):
        super().__init__(name=name,
                         **kwargs)

    @abc.abstractmethod
    def coded_sample(self,
                     target: tfd.Distribution,
                     coder: tfd.Distribution,
                     seed: tf.int64) -> Tuple[int, tf.Tensor]:
        """
        Takes two target distributions and a seed and performs a coded sampling procedure.

        :param target:
        :param coder:
        :param seed:
        :return: Tuple containing the sample index as its first element and the sample itself as the second element
        """

    @abc.abstractmethod
    def decode_sample(self,
                      coder: tfd.Distribution,
                      sample_index: tf.int64,
                      seed: tf.int64) -> tf.Tensor:
        """
        Takes a coding distribution, a sample index and a seed and returns the sample with the given index.

        :param coder:
        :param sample_index:
        :param seed:
        :return:
        """


class ImportanceSampler(Sampler):

    def __init__(self, alpha=np.inf, name="importance_sampler", **kwargs):
        super().__init__(name=name,
                         **kwargs)

        self.alpha = alpha

    def coded_sample(self,
                     target: tfd.Distribution,
                     coder: tfd.Distribution,
                     seed: tf.int64) -> Tuple[int, tf.Tensor]:
        return encode_gaussian_importance_sample(t_loc=target.loc,
                                                 t_scale=target.scale,
                                                 p_loc=coder.loc,
                                                 p_scale=coder.scale,
                                                 seed=seed,
                                                 alpha=self.alpha)

    def decode_sample(self,
                      coder: tfd.Distribution,
                      sample_index: tf.int64,
                      seed: tf.int64) -> tf.Tensor:
        return decode_gaussian_importance_sample(p_loc=coder.loc,
                                                 p_scale=coder.scale,
                                                 index=sample_index,
                                                 seed=seed)


class RejectionSampler(Sampler):

    def __init__(self, sample_buffer_size, r_buffer_size, use_pseudo_sampler=False, name="rejection_sampler", **kwargs):
        super().__init__(name=name,
                         **kwargs)

        self.sample_buffer_size = sample_buffer_size
        self.r_buffer_size = r_buffer_size
        if use_pseudo_sampler:
            self.sample_generator = PseudoSampleGenerator(sample_buffer_size)
        else:
            self.sample_generator = NaiveSampleGenerator(sample_buffer_size)

        # Counts over how many batch elements we averaged over
        self.average_count = tf.Variable(tf.constant(0., dtype=tf.float64),
                                         name="average_count",
                                         trainable=False)
        self._initialized = tf.Variable(False,
                                        name="coder_initialized",
                                        trainable=False)
        self.acceptance_probabilities = tf.Variable(tf.zeros((r_buffer_size, ), dtype=tf.float64),
                                                    name='acceptance_probabilities',
                                                    trainable=False)
        self.spillover_probability = tf.Variable(tf.constant(0., dtype=tf.float64),
                                                 name='spillover_probability',
                                                 trainable=False)
        self.spillover_acceptance_probability = tf.Variable(tf.constant(0., dtype=tf.float64),
                                                            name='spillover_acceptance_probability',
                                                            trainable=False)

    def update(self,
               target: tfd.Distribution,
               coder: tfd.Distribution):
        log_ratios, t_mass, p_mass = get_t_p_mass(target, coder)
        _, pstar_buffer = get_r_pstar(log_ratios, t_mass, p_mass, self.r_buffer_size, dtype=tf.float64)
        acceptance_probabilities = pstar_buffer - tf.concat((tf.constant([0.], dtype=tf.float64), pstar_buffer[:-1]),
                                                            axis=0)
        self.acceptance_probabilities.assign((self.acceptance_probabilities * self.average_count +
                                              acceptance_probabilities) / (self.average_count + 1.))
        self.average_count.assign(self.average_count + 1)
        self.spillover_probability.assign(1. - tf.reduce_sum(self.acceptance_probabilities))
        self.spillover_acceptance_probability.assign(self.acceptance_probabilities[-1] /
                                                     (1. - tf.reduce_sum(self.acceptance_probabilities[:-1])))
        self._initialized.assign(True)

    def get_codelength(self, index):
        assert self._initialized

        if index < self.r_buffer_size:
            return -tf.math.log(self.acceptance_probabilities[index])
        else:
            return -(tf.math.log(self.spillover_probability) +
                     tf.math.log(1. - self.spillover_acceptance_probability) * (index - self.r_buffer_size) +
                     tf.math.log(self.spillover_acceptance_probability))

    def coded_sample(self,
                     target: tfd.Distribution,
                     coder: tfd.Distribution,
                     seed: tf.int64) -> Tuple[int, tf.Tensor]:

        return gaussian_rejection_sample_small(t_dist=target,
                                               p_dist=coder,
                                               sample_buffer_size=self.sample_buffer_size,
                                               r_buffer_size=self.r_buffer_size,
                                               sample_generator=self.sample_generator,
                                               seed=seed)

    def decode_sample(self,
                      coder: tfd.Distribution,
                      sample_index: tf.int64,
                      seed: tf.int64) -> tf.Tensor:
        return self.sample_generator.generate_index(sample_index % self.sample_buffer_size,
                                                    coder,
                                                    seed=seed + sample_index // self.sample_buffer_size)
