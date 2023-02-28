#!/bin/bash

gpu=$1
manual_seed=10
rho=0.2


python main.py \
	--dset=cifar10 \
	--dset_path=/home/Datasets/cifar10 \
	--arch=VGG16 \
	--config_path=./configs/cifar10_big_models_cram_plus_k95_sparse_grad.yaml \
	--workers=2 \
	--epochs=180 \
	--batch_size=128 \
	--gpus=${gpu} \
	--manual_seed=${manual_seed} \
	--experiment_root_path "./experiments/experiments_cram_cifar10_big_models" \
	--exp_name=cifar10_vgg16_cram_plus_k95_rho${rho}_sparse_grad \
	--sam_rho=${rho}   \
	--wandb_group=cifar10_vgg16_cram_plus_k0.95_sparse_grad \
	--wandb_name=CrAM_rho${rho} \
	--wandb_project "cram_cifar10_RN18_big"


