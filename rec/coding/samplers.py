import abc
from typing import Tuple, List

import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np

from .importance_sampling import encode_gaussian_importance_sample, decode_gaussian_importance_sample
from .rejection_sampling import gaussian_rejection_sample_small

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

    def __init__(self, sample_buffer_size, R_buffer_size, name="rejection_sampler", **kwargs):
        super().__init__(name=name,
                         **kwargs)

        self.sample_buffer_size = sample_buffer_size
        self.R_buffer_size = R_buffer_size

    def coded_sample(self,
                     target: tfd.Distribution,
                     coder: tfd.Distribution,
                     seed: tf.int64) -> Tuple[int, tf.Tensor]:

        return gaussian_rejection_sample_small(t_dist=target,
                                               p_dist=coder,
                                               sample_buffer_size=self.sample_buffer_size,
                                               R_buffer_size=self.R_buffer_size,
                                               seed=seed)

    def decode_sample(self,
                      coder: tfd.Distribution,
                      sample_index: tf.int64,
                      seed: tf.int64) -> tf.Tensor:
        return None

