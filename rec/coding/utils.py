import tensorflow as tf


class CodingError(Exception):
    """
    Basis exception class for errors occurring in rec.coding.
    """


def stateless_gumbel_sample(shape, seed):

    return -tf.math.log(-tf.math.log(tf.random.stateless_normal(shape, seed)))
