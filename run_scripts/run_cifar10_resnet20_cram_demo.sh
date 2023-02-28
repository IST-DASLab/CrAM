#!/bin/bash

gpu=$1
manual_seed=10
rho=0.2



python main.py \
	--dset=cifar10 \
	--dset_path=/home/Datasets/cifar10 \
	--arch=resnet20 \
	--config_path=./configs/cifar10_resnet20_cram_plus_multi_spgrad.yaml \
	--workers=2 \
	--epochs=200 \
	--batch_size=128 \
	--checkpoint_freq=20 \
	--gpus=${gpu} \
	--manual_seed=${manual_seed} \
	--experiment_root_path "./experiments/experiments_cram_cifar10" \
	--exp_name=cifar10_resnet20_dense_cram_plus_multi_rho${rho}_sparse_grad \
	--sam_rho=${rho}   \
	--wandb_group=cifar10_resnet20_dense_cram_plus_multi_sparse_grad \
	--wandb_name=CrAM_rho${rho} \
	--wandb_project "cram_cifar10_sparse_grad"


