import abc

import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np

from tqdm import trange

from rec.coding.utils import CodingError
from rec.coding.samplers import Sampler, RejectionSampler, ImportanceSampler

tfl = tf.keras.layers
tfd = tfp.distributions


def sigmoid_inverse(x):
    if tf.reduce_any(x < 0.) or tf.reduce_any(x > 1.):
        raise ValueError(f"x = {x} was not in the sigmoid function's range ([0, 1])!")
    x = tf.clip_by_value(x, 1e-10, 1 - 1e-10)

    return tf.math.log(x) - tf.math.log(1. - x)


class Coder(tfl.Layer, abc.ABC):

    def __init__(self,
                 block_size=None,
                 name="encoder",
                 **kwargs):
        super().__init__(name=name,
                         **kwargs)

        self.block_size = block_size

    def split(self, *args, seed=42):
        """
        Splits the arguments into conformal blocks
        :return:
        """

        tensor_shape = args[0].shape
        num_tensors = len(args)

        flattened = []

        # Check if the shapes are alright
        for tensor in args:
            if tensor.shape != tensor_shape:
                raise CodingError("All tensor arguments supplied to split "
                                  "must have the same batch dimensions!")

            flattened.append(tf.reshape(tensor, [-1]))

        # Total number of dimensions for each tensor
        num_dims = flattened[0].shape[0]

        # We will permute the indices and gather using them to ensure that every block is
        # shuffled the same way
        tf.random.set_seed(seed)
        indices = tf.range(num_dims, dtype=tf.int64)
        indices = tf.random.shuffle(indices)[:, None]

        # Shuffle each tensor the same way
        flattened = [tf.gather_nd(flat, indices) for flat in flattened]

        # Split tensors into blocks

        # Calculate the number of blocks
        num_blocks = num_dims // self.block_size
        num_blocks += (0 if num_dims % self.block_size == 0 else 1)

        all_blocks = []
        for tensor in flattened:

            blocks = []
            for i in range(0, num_dims, self.block_size):
                # The minimum ensures that we do get indices out of bounds
                blocks.append(tensor[i:min(i + self.block_size, num_dims)])

            all_blocks.append(blocks)

        return all_blocks

    def merge(self, *args, shape=None, seed=42):
        """
        Inverse operation to split
        :return:
        """

        if shape is None:
            raise CodingError("Shape cannot be None!")

        # We first merge the blocks back
        tensors = [tf.concat(blocks, axis=0) for blocks in args]

        # Check that all tensors have the same shape now
        num_dims = tensors[0].shape[0]

        for tensor in tensors:
            if tf.rank(tensor) != 1:
                raise CodingError("All supplied tensors to merge must be rank 1!")

            if tensor.shape[0] != num_dims:
                raise CodingError("All tensors must have the same number of dimensions!")

        # We will inverse permute the indices and gather using them
        # to ensure that every block is un-shuffled the same way
        tf.random.set_seed(seed)
        indices = tf.range(num_dims, dtype=tf.int64)
        indices = tf.random.shuffle(indices)
        indices = tf.math.invert_permutation(indices)[:, None]

        tensors = [tf.gather_nd(tensor, indices)
                   for tensor in tensors]

        # Reshape each tensor appropriately
        tensors = [tf.reshape(tensor, shape) for tensor in tensors]

        return tensors

    @abc.abstractmethod
    def encode(self, target_dist, coding_dist, seed, **kwargs):
        pass

    @abc.abstractmethod
    def decode(self, coding_dist, indices, seed, **kwargs):
        pass

    @abc.abstractmethod
    def encode_block(self, target_dist, coding_dist, seed, **kwargs):
        pass

    @abc.abstractmethod
    def decode_block(self, coding_dist, indices, seed, **kwargs):
        pass


def get_auxiliary_coder(coder, auxiliary_var):
    auxiliary_coder = tfd.Normal(loc=tf.zeros_like(coder.loc), scale=tf.sqrt(auxiliary_var))

    return auxiliary_coder


def get_auxiliary_target(target, coder, auxiliary_var):
    coder_var = tf.math.pow(coder.scale, 2)
    target_var = tf.math.pow(target.scale, 2)
    auxiliary_target_mean = (target.loc - coder.loc) * auxiliary_var / coder_var
    auxiliary_target_var = target_var * tf.math.pow(auxiliary_var, 2) / tf.math.pow(coder_var, 2) \
                           + auxiliary_var * (coder_var - auxiliary_var) / coder_var
    ta = tfp.distributions.Normal(loc=auxiliary_target_mean, scale=tf.sqrt(auxiliary_target_var))
    return ta


def get_conditional_coder(coder, auxiliary_var, auxiliary_sample):
    coder_var = tf.math.pow(coder.scale, 2)

    return tfp.distributions.Normal(coder.loc + auxiliary_sample, tf.sqrt(coder_var - auxiliary_var))


def get_conditional_target(target, coder, auxiliary_var, auxiliary_sample):
    coder_var = tf.math.pow(coder.scale, 2)
    target_var = tf.math.pow(target.scale, 2)
    new_t_mean = (coder.loc + (auxiliary_sample * target_var * coder_var +
                               (target.loc - coder.loc) * (coder_var - auxiliary_var) * coder_var) /
                  (target_var * auxiliary_var + coder_var * (coder_var - auxiliary_var)))
    new_t_var = target_var * coder_var * (coder_var - auxiliary_var) / \
                (auxiliary_var * target_var + coder_var * (coder_var - auxiliary_var))
    return tfp.distributions.Normal(new_t_mean, tf.sqrt(new_t_var))


class GaussianCoder(Coder):
    def __init__(self,
                 kl_per_partition,
                 sampler: Sampler,
                 block_size=None,
                 name="gaussian_encoder",
                 **kwargs):

        super().__init__(name=name,
                         block_size=block_size,
                         **kwargs)

        # ---------------------------------------------------------------------
        # Assign parameters
        # ---------------------------------------------------------------------
        self.sampler = sampler

        self.kl_per_partition = tf.cast(kl_per_partition, tf.float32)

        # ---------------------------------------------------------------------
        # Create parameters for the auxiliary variables
        # ---------------------------------------------------------------------

        # The auxiliary variables are always scaled w.r.t the coding distribution, i.e.
        # Var[A_i] = R_i * Var_{Z_i ~ P(Z_i)}[Z_i]
        # The variance ratio at index i creates a chunk that has KL divergence 1/(i+1) times the overall KL divergence
        self.aux_variable_variance_ratios = tf.Variable(tf.constant([1.], dtype=tf.float32),
                                                        shape=tf.TensorShape([None]),
                                                        name="aux_variable_variance_ratios",
                                                        trainable=False)

        # Counts over how many batch elements we averaged over
        self.average_counts = tf.Variable(tf.constant([1.], dtype=tf.float32),
                                          shape=tf.TensorShape([None]),
                                          name="average_counts",
                                          trainable=False)

        self._initialized = tf.Variable(False,
                                        name="coder_initialized",
                                        trainable=False)

    def update_auxiliary_variance_ratios(self,
                                         target_dist,
                                         coding_dist,
                                         seed=42,
                                         **kwargs):

        print(f"Updating {self.name}!")
        if self.block_size is None:
            self.update_block_auxiliary_variance_ratios(target_dist,
                                                        coding_dist,
                                                        **kwargs)

        else:
            # We split the distributions into blocks, and then batch them
            target_loc, target_scale, coding_loc, coding_scale = self.split(target_dist.loc,
                                                                            target_dist.scale,
                                                                            coding_dist.loc,
                                                                            coding_dist.scale,
                                                                            seed=seed)

            # We leave off the last block, because its size might be different to the rest
            block_target = tfd.Normal(loc=tf.stack(target_loc[:-1], axis=0),
                                      scale=tf.stack(target_scale[:-1], axis=0))

            block_coder = tfd.Normal(loc=tf.stack(coding_loc[:-1], axis=0),
                                     scale=tf.stack(coding_scale[:-1], axis=0))

            self.update_block_auxiliary_variance_ratios(
                target_dist=block_target,
                coding_dist=block_coder,
                **kwargs
            )

    def update_block_auxiliary_variance_ratios(self,
                                               target_dist,
                                               coding_dist,
                                               relative_tolerance=1e-4,
                                               max_iters=10000,
                                               learning_rate=0.001):
        # Gather distribution statistics
        target_loc = target_dist.loc
        target_scale = target_dist.scale

        coding_loc = coding_dist.loc
        coding_scale = coding_dist.scale

        data_dims = list(range(1, tf.rank(target_loc)))

        # The first dimension is the "batch" dimension, so we preserve it
        total_kl = tf.reduce_sum(tfd.kl_divergence(target_dist, coding_dist), axis=data_dims)
        # Calculate the number of required auxiliary variables for each batch element
        num_aux_variables = 1 + tf.cast(tf.math.floor(total_kl / self.kl_per_partition), tf.int32)
        max_num_variables = tf.reduce_max(num_aux_variables)
        # get dynamic shape
        current_max = tf.shape(self.aux_variable_variance_ratios)[0]

        if max_num_variables > current_max:
            aux_variable_variance_ratios_copy = tf.identity(self.aux_variable_variance_ratios)
            self.aux_variable_variance_ratios = tf.Variable(tf.zeros((max_num_variables,), dtype=tf.float32),
                                                            shape=tf.TensorShape([None]),
                                                            name="sum_averaged_variance_ratios",
                                                            trainable=False)
            self.aux_variable_variance_ratios[:current_max].assign(aux_variable_variance_ratios_copy)

            average_counts_copy = tf.identity(self.average_counts)
            self.average_counts = tf.Variable(tf.zeros((max_num_variables,), dtype=tf.float32),
                                              shape=tf.TensorShape([None]),
                                              name="average_counts",
                                              trainable=False)
            self.average_counts[:current_max].assign(average_counts_copy)

        # Perform initialization for each possible KL ratio
        for ratio in range(max_num_variables, 1, -1):

            # We will only update the distributions with high enough KL
            indices = tf.where(num_aux_variables >= ratio)

            # Number of current batch elements
            num_elements = indices.shape[0]

            # Create dummy distributions
            target = tfd.Normal(loc=tf.gather_nd(target_loc, indices),
                                scale=tf.gather_nd(target_scale, indices))

            coder = tfd.Normal(loc=tf.gather_nd(coding_loc, indices),
                               scale=tf.gather_nd(coding_scale, indices))

            # Update KL
            total_kl = tf.reduce_sum(tfd.kl_divergence(target, coder), axis=data_dims)

            # Initialize ratio parameters
            if self.aux_variable_variance_ratios[ratio - 1] > 0.:
                init = self.aux_variable_variance_ratios[ratio - 1]
            elif ratio < max_num_variables:
                init = self.aux_variable_variance_ratios[ratio]
            else:
                init = 1. / ratio

            init = sigmoid_inverse(init)
            reparameterized_aux_variable_var_ratio = tf.Variable(init)

            # Compensate in the learning rate for the increased loss
            optimizer = tf.optimizers.SGD(learning_rate=learning_rate)

            # Optimize the current ratio using SGD
            prev_loss = np.inf

            with trange(max_iters) as progress_bar:
                for _ in progress_bar:
                    with tf.GradientTape() as tape:
                        aux_variable_variance_ratio = tf.nn.sigmoid(reparameterized_aux_variable_var_ratio)

                        auxiliary_variance = aux_variable_variance_ratio * tf.math.pow(coder.scale, 2)
                        aux_target = get_auxiliary_target(target=target,
                                                          coder=coder,
                                                          auxiliary_var=auxiliary_variance)

                        aux_coder = get_auxiliary_coder(coder=coder,
                                                        auxiliary_var=auxiliary_variance)

                        # Get the KL between q(A_i | Z) and p(A_i | Z)
                        auxiliary_kl = tf.reduce_sum(tfd.kl_divergence(aux_target, aux_coder),
                                                     axis=data_dims)

                        # Make a quadratic loss
                        # kl_loss = tf.reduce_mean(tf.math.pow(auxiliary_kl - total_kl / tf.cast(ratio, tf.float32), 2))
                        # kl_loss = tf.reduce_mean(tf.math.pow(total_kl - auxiliary_kl - self.kl_per_partition * (ratio - 1), 2))
                        aux_kl_loss = tf.where(auxiliary_kl > self.kl_per_partition,
                                               tf.math.pow(auxiliary_kl - self.kl_per_partition, 2),
                                               0.)
                        remaining_kl_loss = tf.where(total_kl - auxiliary_kl > self.kl_per_partition * (ratio - 1),
                                                     tf.math.pow((total_kl - auxiliary_kl) - self.kl_per_partition * (
                                                             ratio - 1), 2),
                                                     0.)
                        kl_loss = tf.reduce_mean(aux_kl_loss + remaining_kl_loss)

                    gradient = tape.gradient(kl_loss, reparameterized_aux_variable_var_ratio)
                    optimizer.apply_gradients([(gradient, reparameterized_aux_variable_var_ratio)])

                    # Early stop if the loss decreases less than the tolerance
                    if tf.abs(prev_loss - kl_loss) < relative_tolerance:
                        break

                    prev_loss = kl_loss

                    progress_bar.set_description(f"Ratio {ratio}, {num_elements}/{target_loc.shape[0]} items - "
                                                 f"ratio: {aux_variable_variance_ratio:.4f}, "
                                                 f"avg_aux_kl: {tf.reduce_mean(auxiliary_kl):.3f}+-{tf.math.reduce_std(auxiliary_kl):.3f}, "
                                                 f"avg target_kl: {tf.reduce_mean(total_kl) / ratio:.3f}, "
                                                 f"avg kl: {tf.reduce_mean(total_kl):.3f}, "
                                                 f"loss: {kl_loss:.3f}")

            self.aux_variable_variance_ratios[ratio - 1].assign(
                (self.aux_variable_variance_ratios[ratio - 1] * self.average_counts[ratio - 1] +
                 aux_variable_variance_ratio * num_elements) /
                (self.average_counts[ratio - 1] + num_elements))
            self.average_counts[ratio - 1].assign(self.average_counts[ratio - 1] + num_elements)
            auxiliary_variance = self.aux_variable_variance_ratios[ratio - 1] * tf.math.pow(coder.scale, 2)

            # Once the optimization is finished, calculate the new target and coding distributions
            auxiliary_sample = aux_target.sample()
            target = get_conditional_target(target=target,
                                            coder=coder,
                                            auxiliary_var=auxiliary_variance,
                                            auxiliary_sample=auxiliary_sample)

            coder = get_conditional_coder(coder=coder,
                                          auxiliary_var=auxiliary_variance,
                                          auxiliary_sample=auxiliary_sample)

            # Update the distribution statistics tensors
            target_loc = tf.tensor_scatter_nd_update(target_loc, indices, target.loc)
            target_scale = tf.tensor_scatter_nd_update(target_scale, indices, target.scale)

            coding_loc = tf.tensor_scatter_nd_update(coding_loc, indices, coder.loc)
            coding_scale = tf.tensor_scatter_nd_update(coding_scale, indices, coder.scale)

        self._initialized.assign(True)

    def encode(self, target_dist, coding_dist, seed, **kwargs):

        print(f"Coding sample in {self.name}")
        if self.block_size is None:
            return self.encode_block(target_dist,
                                     coding_dist,
                                     seed,
                                     **kwargs)

        else:
            samp_shape = target_dist.loc.shape

            samples = []
            indices = []

            split_statistics = self.split(target_dist.loc,
                                          target_dist.scale,
                                          coding_dist.loc,
                                          coding_dist.scale,
                                          seed=seed)

            num_blocks = len(split_statistics[0])

            for block_index, \
                (target_loc,
                 target_scale,
                 coding_loc,
                 coding_scale) in enumerate(zip(*split_statistics)):
                print(f"Coding sample in {self.name}, block {block_index + 1}/{num_blocks}!")

                # We add the extra dimension because encode is expecting
                # images in batches of 1.
                ind, samp = self.encode_block(target_dist=tfd.Normal(loc=target_loc[None, :],
                                                                     scale=target_scale[None, :]),
                                              coding_dist=tfd.Normal(loc=coding_loc[None, :],
                                                                     scale=coding_scale[None, :]),
                                              seed=seed,
                                              **kwargs)

                samples.append(samp[0, :])
                indices = indices + ind

            # Note the comma: merge returns a singleton list, which is why it is needed.
            sample, = self.merge(samples, shape=samp_shape, seed=seed)

            return indices, sample

    def decode(self, coding_dist, indices, seed, **kwargs):
        pass

    def encode_block(self, target_dist, coding_dist, seed, update_sampler=False, verbose=False):

        if not self._initialized:
            raise CodingError(
                "Coder has not been initialized yet, please call update_auxiliary_variance_ratios() first!")

        if target_dist.loc.shape[0] != 1:
            raise CodingError("For encoding, batch size must be 1.")

        indices = []

        total_kl = tf.reduce_sum(tfd.kl_divergence(target_dist, coding_dist))
        print('Encoding latent variable with KL={}'.format(total_kl))
        num_aux_variables = tf.cast(tf.math.ceil(total_kl / self.kl_per_partition), tf.int32)

        # If there are more auxiliary variables needed than what we are already storing, we update our estimates
        current_max = tf.shape(self.aux_variable_variance_ratios)[0]
        if num_aux_variables > current_max:
            raise CodingError("KL divergence higher than auxiliary variables can account for. "
                              "Update auxiliary variable ratios with high-enough KL divergence."
                              "Maximum possible KL divergence is {}.".format(
                current_max.numpy() * self.kl_per_partition))

        # We iterate backward until the second entry in ratios. The first entry is 1.,
        # in which case we just draw the final sample.
        for i in range(num_aux_variables - 1, 0, -1):
            aux_variable_variance_ratio = self.aux_variable_variance_ratios[i]
            auxiliary_var = aux_variable_variance_ratio * tf.math.pow(coding_dist.scale, 2)

            auxiliary_target = get_auxiliary_target(target=target_dist,
                                                    coder=coding_dist,
                                                    auxiliary_var=auxiliary_var)

            auxiliary_coder = get_auxiliary_coder(coder=coding_dist,
                                                  auxiliary_var=auxiliary_var)

            if update_sampler:
                self.sampler.update(auxiliary_target, auxiliary_coder)
                auxiliary_sample = auxiliary_target.sample()
                print('Sampler updated')
            else:
                index, auxiliary_sample = self.sampler.coded_sample(target=auxiliary_target,
                                                                    coder=auxiliary_coder,
                                                                    seed=seed)
                if verbose:
                    print(f'Auxiliary sample {i} found at index {index}')
                indices.append(index)
            seed += 1

            target_dist = get_conditional_target(target=target_dist,
                                                 coder=coding_dist,
                                                 auxiliary_var=auxiliary_var,
                                                 auxiliary_sample=auxiliary_sample)

            coding_dist = get_conditional_coder(coder=coding_dist,
                                                auxiliary_var=auxiliary_var,
                                                auxiliary_sample=auxiliary_sample)

        # Sample the last auxiliary variable
        if update_sampler:
            self.sampler.update(target_dist, coding_dist)
            sample = target_dist.sample()
            print('Sampler updated')
        else:
            index, sample = self.sampler.coded_sample(target=target_dist,
                                                      coder=coding_dist,
                                                      seed=seed)
            if verbose:
                print('Auxiliary sample found at index {}'.format(index))
            indices.append(index)

        return indices, sample

    def decode_block(self, coding_dist, indices, seed, **kwargs):
        num_aux_variables = len(indices)

        indices.reverse()
        for i in range(num_aux_variables - 1, 0, -1):
            aux_variable_variance_ratio = self.aux_variable_variance_ratios[i]
            auxiliary_var = aux_variable_variance_ratio * tf.math.pow(coding_dist.scale, 2)

            auxiliary_coder = get_auxiliary_coder(coder=coding_dist,
                                                  auxiliary_var=auxiliary_var)

            auxiliary_sample = self.sampler.decode_sample(coder=auxiliary_coder,
                                                          sample_index=indices[i],
                                                          seed=seed)
            seed += 1

            coding_dist = get_conditional_coder(coder=coding_dist,
                                                auxiliary_var=auxiliary_var,
                                                auxiliary_sample=auxiliary_sample)

        sample = self.sampler.decode_sample(coder=coding_dist,
                                            sample_index=indices[0],
                                            seed=seed)
        return sample

    def get_codelength(self, indicies):
        return sum([self.sampler.get_codelength(i) for i in indicies])
