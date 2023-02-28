#!/bin/bash

gpu=$1
manual_seed=10
rho=0.2


python main.py \
	--dset=cifar10 \
	--dset_path=/home/Datasets/cifar10 \
	--arch=resnet18_cifar_big \
	--config_path=./configs/cifar10_big_models_cram_plus_k95_sparse_grad.yaml \
	--workers=2 \
	--epochs=180 \
	--batch_size=128 \
	--gpus=${gpu} \
	--manual_seed=${manual_seed} \
	--experiment_root_path "./experiments/experiments_cram_cifar10_big_RN18" \
	--exp_name=cifar10_resnet18_big_cram_plus_k95_rho${rho}_sparse_grad \
	--sam_rho=${rho}   \
	--wandb_group=cifar10_resnet18_big_cram_plus_k0.95_sparse_grad \
	--wandb_name=CrAM_rho${rho} \
	--wandb_project "cram_cifar10_RN18_big"


