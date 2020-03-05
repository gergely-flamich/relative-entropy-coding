import abc

import tensorflow as tf
import tensorflow_probability as tfp

from .utils import CodingError

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

    def __init__(self,
                 target_dist,
                 coding_dist,
                 name="gaussian_encoder",
                 **kwargs):

        super().__init__(target_dist=target_dist,
                         coding_dist=coding_dist,
                         name=name,
                         **kwargs)

    def call(self):
        pass
