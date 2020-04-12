# relative-entropy-coding

python examples/train_generative_model.py print_config with model=resnet_vae num_res_blocks=20 tensorboard_log_freq=100 dataset_info.dataset_name=cifar10

Command to train the auxiliary ratios on enigma:
python3 examples/compression_performance.py with mode='initialize' model_save_base_dir='/scratch/mh740/models/relative-entropy-coding' train_dataset='imagenet32' dataset_info.dataset_name='imagenet32' num_res_blocks=24 dataset_info.dataset_base_path='/scratch/mh740/datasets' dataset_info.split=train num_test_images=100



CLIC models (edit in model config):
BPP 0.1  beta_0.500_lamb_0.250_laplace_target_bpp_0.100/
BPP 0.3  beta_1.500_lamb_0.250_laplace/
BPP 1.0 (not ready yet) beta_0.100_lamb_0.250_laplace_target_bpp_1.000/
