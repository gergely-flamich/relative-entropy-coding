import abc

import tensorflow as tf
import tensorflow_probability as tfp

from .utils import CodingError
from .importance_sampling import encode_gaussian_importance_sample

tfl = tf.keras.layers


class Encoder(tfl.Layer, abc.ABC):

    def __init__(self,
                 target_dist,
                 coding_dist,
                 name="encoder",
                 **kwargs):

        super().__init__(name=name,
                         **kwargs)

        self.target_dist = target_dist
        self.coding_dist = coding_dist


class Decoder(tfl.Layer, abc.ABC):

    def __init__(self,
                 coding_dist,
                 name="decoder",
                 **kwargs):

        super().__init__(name=name,
                         **kwargs)

        self.coding_dist = coding_dist


class GaussianEncoder(Encoder):

    AVAILABLE_SAMPLERS = ["importance",
                          "greedy_rejection"]

    def __init__(self,
                 target_dist,
                 coding_dist,
                 sampler="importance",
                 name="gaussian_encoder",
                 **kwargs):

        super().__init__(target_dist=target_dist,
                         coding_dist=coding_dist,
                         name=name,
                         **kwargs)

        if sampler not in self.AVAILABLE_SAMPLERS:
            raise CodingError(f"Sampler must be one of {self.AVAILABLE_SAMPLERS}, but {sampler} was given!")

        self.sampler = sampler

    def get_auxiliary_distribution(self, aux_):
        pass

    def get_partitions(self, kl_per_partition):
        return []

    def call(self, seed, kl_per_partition=10.):

        # Get target and coding distribution statistics per partition
        means, vars = self.get_partitions(kl_per_partition=kl_per_partition)

        for (t_loc, p_loc), (t_scale, p_scale) in zip(means, vars):

            if self.sampler == "importance":
                sample, index = encode_gaussian_importance_sample(t_loc,
                                                                  t_scale,
                                                                  p_loc,
                                                                  p_scale,
                                                                  seed)

            else:
                raise NotImplementedError

