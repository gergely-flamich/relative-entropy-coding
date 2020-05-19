# -*- coding: utf-8 -*-
"""HiLLoC RVAE Compression

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11967hjFQczjW21cLLTFhOnTurx3mSBVD
"""
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as onp
params = onp.load('rvae_weights_24layer.npy', allow_pickle=True).item()
image = onp.load('sample_cifar_image.npy', allow_pickle=True)
#
# import scipy
#
# def sigmoid(x):
#   return 1 / (1 + np.exp(-x))
#
# import jax
# from jax import np, lax, random, jit, nn
# from jax.scipy.stats import norm
# from jax.nn.initializers import glorot_normal, normal, ones
# import itertools
# from functools import partial
# from autograd.builtins import tuple as ag_tuple
#
#
# from datasets import load_dataset
# import tensorflow as tf
# # import tensorflow_probability as tfp
# # tfd = tfp.distributions
#
# dataset, _ = load_dataset(
#   tfds_name="downsampled_imagenet/32x32:1.0.0",
#   dataset_path="/scratch/mh740/datasets/imagenet32",
#   num_pixels=32 * 32,
#   split="test",
#   normalizer=256.,
#   training_patch_size=None,
#   test_split_name="validation")
# dataset = dataset.take(1000)
#
#
def get_params(name, layer=None):
  """Interface for the dict of tensorflow params"""
  if layer is not None:
    s = 'model/IAF_0_{}/{}/'.format(layer, name)
  else:
    s = 'model/{}/'.format(name)
  W = params[s+'V:0']
  g = params[s+'g:0']
  b = params[s+'b:0']
  return W, g, b
#
#
#
# z_size = 32
# h_size = 160
# depth = 24
#
# dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
# one = (1, 1)
#
# conv = partial(lax.conv_general_dilated,
#                dimension_numbers=dimension_numbers,
#                padding='SAME')
#
# def l2_normalize(x, axis, epsilon=1e-12):
#   l2 = np.sum(np.square(x), axis=axis, keepdims=True)
#   l2 = np.sqrt(np.maximum(l2, epsilon))
#   return x / l2
#
# def apply_conv(params, inputs, strides=one):
#     W, g, b = params
#     W = np.exp(g) * l2_normalize(W, axis=[0, 1, 2])  # weight norm
#     return conv(inputs, W, strides) + b
#
# def empty_list(l):
#   return [None for i in range(l)]
#
# def get_params(name, layer=None):
#   if layer is not None:
#     s = 'model/IAF_0_{}/{}/'.format(layer, name)
#   else:
#     s = 'model/{}/'.format(name)
#   W = params[s+'V:0']
#   g = params[s+'g:0']
#   b = params[s+'b:0']
#   return W, g, b
#
#
# ## ------------------------- MODEL ---------------------------------
#
# def down_split(layer, inputs):
#   out = nn.elu(inputs)
#   out = apply_conv(get_params('down_conv1', layer), out)
#   prior_mean, prior_logstd, rz_mean, rz_logstd, _, h_det = \
#     np.split(out, [z_size, 2*z_size, 3*z_size, 4*z_size,
#                    4*z_size + h_size], axis=-1)
#   return (prior_mean, prior_logstd, rz_mean, rz_logstd), h_det
#
# def down_merge(layer, h_det, inputs, z):
#   h = np.concatenate([z, h_det], axis=-1)
#   h = nn.elu(h)
#   h = apply_conv(get_params('down_conv2', layer), h)
#   inputs = inputs + 0.1 * h
#   return inputs
#
# def up_pass(inputs):
#
#   inputs = apply_conv(get_params('x_enc'), inputs, strides=(2, 2))
#
#   q_stats = empty_list(depth)  # these are all we care about from the up pass
#   for i in range(depth):
#     # up split
#     out = nn.elu(inputs)
#     out = apply_conv(get_params('up_conv1', i), out)
#     qz_mean, qz_logstd, _, h = np.split(out, [z_size, 2*z_size, 2*z_size + h_size], axis=-1)
#     q_stats[i] = qz_mean, qz_logstd
#
#     # up merge
#     h = nn.elu(h)
#     h = apply_conv(get_params('up_conv3', i), h)
#     inputs = inputs + 0.1 * h
#   return q_stats
#
# def upsample_and_postprocess(inputs):
#   out = nn.elu(inputs)
#   W, g, b = get_params('x_dec')
#   # W is HWOI rather than HWIO, and gets normalised incorrectly it seems
#   W = np.exp(g).reshape(1, 1, -1, 1) * l2_normalize(W, axis=(0, 1, 2))
#   # transpose_kernel for compatibility with tf conv transpose
#   out = lax.conv_transpose(out, W, dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
#                            strides=(2, 2), padding='SAME',
#                            transpose_kernel=True)
#   out += b
#   return np.clip(out, -0.5 + 1 / 512., 0.5 - 1 / 512.)
#
#
# from rec.models.resnet_vae import BidirectionalResNetVAE
# model = BidirectionalResNetVAE(
#   num_res_blocks=24,
#   sampler='beam_search',
#   sampler_args={'n_beams': 10, 'extra_samples': 1.1},
#   coder_args={},
#   likelihood_function="discretized_logistic",
#   learn_likelihood_scale=True,
#   first_kernel_size=(5, 5),
#   first_strides=(2, 2),
#   kernel_size=(3, 3),
#   strides=(1, 1),
#   deterministic_filters=160,
#   stochastic_filters=32,
#   use_iaf=False,
#   kl_per_partition=8.,
#   latent_size="variable",
#   ema_decay=0.999,
#   name="resnet_vae")
#
# # Initialize_model_weights
# model(tf.zeros((1, 32, 32, 3)))

def load_weights(model, params):
  def get_var(model, name):
    for v in model.variables:
      if name in v.name:
        return v
  def assign_layer(layer, param_tuple, weird=False):
    W, g, b = param_tuple
    get_var(layer, 'kernel_weights').assign(W)
    if weird:
      get_var(layer, 'kernel_log_scale').assign(g[None, None, :, None])
    else:
      get_var(layer, 'kernel_log_scale').assign(g[None, None, None, :])
    get_var(layer, 'bias').assign(b)

  def assign_block(block, j):
    W, g, b = get_params('up_conv1', j)
    W_qz_mean, W_qz_std, _, W_h = onp.split(W, [32, 2*32, 2*32 + 160], axis=-1)
    g_qz_mean, g_qz_std, _, g_h = onp.split(g, [32, 2*32, 2*32 + 160], axis=-1)
    b_qz_mean, b_qz_std, _, b_h = onp.split(b, [32, 2*32, 2*32 + 160], axis=-1)
    assign_layer(block.infer_posterior_loc_head, (W_qz_mean, g_qz_mean, b_qz_mean))
    assign_layer(block.infer_posterior_log_scale_head, [W_qz_std, g_qz_std, b_qz_std])
    if not j == 23:
      assign_layer(block.infer_conv1, [W_h, g_h, b_h])
      assign_layer(block.infer_conv2, get_params('up_conv3', j))


    W, g, b = get_params('down_conv1', j)
    W_p_mean, W_p_std, W_z_mean, W_z_std, _, W_h = onp.split(W, [32, 2*32, 3*32, 4*32, 4*32 + 160], axis=-1)
    g_p_mean, g_p_std, g_z_mean, g_z_std, _, g_h = onp.split(g, [32, 2*32, 3*32, 4*32, 4*32 + 160], axis=-1)
    b_p_mean, b_p_std, b_z_mean, b_z_std, _, b_h = onp.split(b, [32, 2*32, 3*32, 4*32, 4*32 + 160], axis=-1)

    assign_layer(block.prior_loc_head, (W_p_mean, g_p_mean, b_p_mean))
    assign_layer(block.prior_log_scale_head, (W_p_std, g_p_std, b_p_std))
    assign_layer(block.gen_posterior_loc_head, (W_z_mean, g_z_mean, b_z_mean))
    assign_layer(block.gen_posterior_log_scale_head, (W_z_std, g_z_std, b_z_std))
    assign_layer(block.gen_conv1, (W_h, g_h, b_h))
    assign_layer(block.gen_conv2, get_params('down_conv2', j))




  assign_layer(model.first_infer_conv, get_params('x_enc'))
  for j in range(24):
    assign_block(model.residual_blocks[j], 23-j)

  assign_layer(model.last_gen_conv, get_params('x_dec'), weird=True)
  model._generative_base.assign(params['model/h_top:0'])
  model.likelihood_log_scale.assign(params['model/dec_log_stdv:0'])
#
# load_weights(model, params)
#
#
# cumulative_log_likelihood = 0.
# cumulative_kld = 0.
# our_cumulative_log_likelihood = 0.
# our_cumulative_kld = 0.
#
# for ind, ds_image in enumerate(dataset):
#   model(ds_image[None, :])
#   # print(ds_image)
#   #
#   our_cumulative_log_likelihood += np.sum(model.log_likelihood.numpy())
#   our_cumulative_kld += np.sum(model.kl_divergence(empirical=False).numpy())
#   image = ds_image[None, :].numpy()
#
#   # image = np.clip((image.astype('float64') + 0.5) / 256.0, 0.0, 1.0) - 0.5
#   _, h, w, c = image.shape
#   z_view = lambda head: head[0]
#   x_view = lambda head: head[1]
#
#   likelihood_scale = np.exp(params['model/dec_log_stdv:0'])
#   h_init = params['model/h_top:0']
#
#   q_stats = up_pass(image)
#
#   # run down pass and pop according to posterior, top down
#   inputs = np.tile(np.reshape(h_init, (1, 1, 1, -1)),
#                    (1, h // 2, w // 2, 1))  # assuming batch size 1 for now
#   zs = empty_list(depth)
#   for i in reversed(range(depth)):
#     (prior_mean, prior_logstd, rz_mean, rz_logstd), h_det = \
#       down_split(i, inputs)
#     qz_mean, qz_logstd = q_stats[i]
#     qz_mean, qz_logstd = q_stats[i]
#
#     post_mean = qz_mean + rz_mean
#     post_std = np.exp(qz_logstd + rz_logstd)
#     prior_mean = prior_mean
#     prior_std = np.exp(prior_logstd)
#     z = post_mean + post_std * onp.random.randn(*post_mean.shape)
#     cumulative_kld += np.sum(scipy.stats.norm.logpdf(z, post_mean, post_std))
#     cumulative_kld -= np.sum(scipy.stats.norm.logpdf(z, prior_mean, prior_std))
#     inputs = down_merge(i, h_det, inputs, z)
#
#   # push data
#   x_mean = upsample_and_postprocess(inputs)
#   binsize = 1. / 256.
#   discretized_input = np.floor(image / binsize) * binsize
#   discretized_input = (discretized_input - x_mean) / likelihood_scale
#
#   log_likelihood = sigmoid(discretized_input + binsize / likelihood_scale)
#   log_likelihood = log_likelihood - sigmoid(discretized_input)
#
#   log_likelihood = np.log(log_likelihood + 1e-7)
#   cumulative_log_likelihood += np.sum(log_likelihood)
#
#   # print(cumulative_log_likelihood)
#   # print(cumulative_kld / (ind+1))
#   print(1.44 * (-cumulative_log_likelihood + cumulative_kld) / ((ind+1) * 3 * 32 * 32))
#   print(1.44 * (-our_cumulative_log_likelihood + our_cumulative_kld) / ((ind+1) * 3 * 32 * 32))
#
#
#
