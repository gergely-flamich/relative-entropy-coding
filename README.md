# relative-entropy-coding

python examples/train_generative_model.py print_config with model=resnet_vae num_res_blocks=20 tensorboard_log_freq=100 dataset_info.dataset_name=cifar10

Command to train the auxiliary ratios on enigma:
python3 examples/compression_performance.py with mode='initialize' model_save_base_dir='/scratch/mh740/models/relative-entropy-coding' train_dataset='imagenet32' dataset_info.dataset_name='imagenet32' num_res_blocks=24 dataset_info.dataset_base_path='/scratch/mh740/datasets' dataset_info.split=train num_test_images=100

