import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot as plt

from rejection_sampling import *

def get_t_p_gauss(filename, dims=1000):
    t_mean = np.tile(np.load(filename + 'post_loc.npy')[0], 1 + dims // 50)[:dims]
    t_scale = np.tile(np.load(filename + 'post_scale.npy')[0], 1 + dims // 50)[:dims]
    p_mean = np.tile(np.load(filename + 'prior_loc.npy'), 1 + dims // 50)[:dims]
    p_scale = np.tile(np.load(filename + 'prior_scale.npy'), 1 + dims // 50)[:dims]

    ndims = p_mean.shape[0]
    print('Dimensions: {}'.format(ndims))
    p = tfp.distributions.Normal(loc=p_mean, scale=p_scale)
    t = tfp.distributions.Normal(loc=t_mean, scale=t_scale)
    print('KL divergence: {}'.format(tf.reduce_sum(tfp.distributions.kl_divergence(t, p))))
    return t, p

# tf.debugging.set_log_device_placement(True)
mnist_path = '/scratch/mh740/mnist_posteriors/beta_1_latents_50/test/img_{}/'
#
# print(encode_gaussian_rejection_sample(t, p, buffer_size=1000000))

t_list = []
p_list = []
for i in range(20):
    t, p = get_t_p_gauss(filename=mnist_path.format(i), dims=500)
    t_list.append(t)
    p_list.append(p)

target_kl = 10.
# aux_ratios = np.array([0.68339353, 0.54674925, 0.46513237, 0.4246385,  0.39013079, 0.37761325])
# aux_ratios = preprocessing_auxiliary_ratios(t_list, p_list, target_kl)
# np.savetxt('aux_ratios.txt', aux_ratios)
aux_ratios = np.loadtxt('aux_ratios.txt')
print(aux_ratios)
# plt.plot(aux_ratios)
# plt.show()


t, p = get_t_p_gauss(filename=mnist_path.format(333), dims=30)
preprocessing_probs([t], [p], buffer_size=10000000)


print(gaussian_rejection_sample_large(t, p, target_kl, aux_ratios, 10000, 10000000))
