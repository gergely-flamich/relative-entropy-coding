import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

kl_list = ['2.0', '3.0', '4.0', '5.0', '6.0']
es_list = ['1.', '1.1', '1.2', '1.5']
nb_list = ['1', '10', '50']

file_path = '/scratch/mh740/models/relative-entropy-coding/imagenet32/resnet_vae/gaussian/blocks_24/beta_1.000_lamb_0.100/compressor_initialized_{}/results_{}_beam_{}_es.csv'
fig, axes = plt.subplots(3, len(kl_list), figsize=(20, 20))
for k, kl in enumerate(kl_list):
    mask = np.ones((len(es_list), len(nb_list)), dtype=np.int32)
    crash_mask = np.ones((len(es_list), len(nb_list)), dtype=np.int32)
    overhead = np.zeros((len(es_list), len(nb_list)), dtype=np.float32)
    time = np.zeros((len(es_list), len(nb_list)), dtype=np.float32)
    crashes = np.zeros((len(es_list), len(nb_list)), dtype=np.int32)
    for i, es in enumerate(es_list):
        for j, nb in enumerate(nb_list):
            try:
                with open(file_path.format(kl, nb, es), "r") as results:
                    line_list = [line.split(',') for line in results]
                if len(line_list) < 11:
                    continue
                n_crash = sum([1 if float(l[9]) == -1.0 else 0 for l in line_list[1:]])
                crashes[i, j] = n_crash
                if n_crash == 10:
                    crash_mask[i, j] = 0
                    continue
                avg_times = sum([float(l[9]) if float(l[9]) != -1.0 else 0. for l in line_list[1:]]) / (10 - n_crash)
                avg_elbo = sum([float(l[0]) + float(l[1]) if float(l[9]) != -1.0 else 0. for l in line_list[1:]]) / (10 - n_crash)
                avg_codelength = sum([float(l[4]) + float(l[5]) if float(l[9]) != -1.0 else 0. for l in line_list[1:]]) / (10 - n_crash)
                # print(n_crash)
                # print(avg_elbo)
                # print([float(l[0]) + float(l[1]) if float(l[9]) != -1.0 else 0. for l in line_list[1:]])
                overhead[i, j] = -1. + avg_codelength / avg_elbo
                time[i, j] = avg_times
                mask[i, j] = 0
                crash_mask[i, j] = 0
            except FileNotFoundError:
                continue

    sns.heatmap(overhead.transpose(), annot=True, mask=mask.transpose(), ax=axes[0, k], vmin=0, vmax=1., xticklabels=es_list, yticklabels=nb_list, fmt='.3f')
    axes[0, k].set_title('Overhead \n KL per partition={}'.format(kl))
    axes[0, k].set_xlabel('Oversampling')
    axes[0, k].set_ylabel('N beams')
    sns.heatmap(time.transpose(), annot=True, mask=mask.transpose(), ax=axes[1, k], vmin=0, vmax=1000, xticklabels=es_list, yticklabels=nb_list, fmt='.1f')
    axes[1, k].set_title('Runtime \n KL per partition={}'.format(kl))
    axes[1, k].set_xlabel('Oversampling')
    axes[1, k].set_ylabel('N beams')
    sns.heatmap(crashes.transpose(), annot=True, mask=crash_mask.transpose(), ax=axes[2, k], vmin=0, vmax=10, xticklabels=es_list, yticklabels=nb_list)
    axes[2, k].set_title('Number of crashes \n KL per partition={}'.format(kl))
    axes[2, k].set_xlabel('Oversampling')
    axes[2, k].set_ylabel('N beams')
plt.savefig('examples/figures/hyperparam_tuning.pdf')
