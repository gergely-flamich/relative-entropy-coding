import abc
import tensorflow as tf


class LossyCompressionModel(tf.keras.Model, abc.ABC):

    def __init__(self,
                 name="lossy_compression_model",
                 **kwargs):

        super().__init__(name=name, **kwargs)

    @abc.abstractmethod
    def compress(self, file_path, image, seed, sampler, block_size, max_index):
        pass

    @abc.abstractmethod
    def decompress(self, file_path, sampler):
        pass
