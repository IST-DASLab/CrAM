#!/bin/bash

gpu=0,1,2,3
manual_seed=10
rho=0.05


python main.py \
	--dset=imagenet \
	--dset_path=$IMAGENET_PATH \
	--arch=resnet50 \
	--config_path=./configs/imagenet_resnet50_cram_NM_plus_spgrad.yaml \
	--workers=40 \
	--checkpoint_freq=10 \
	--epochs=100 \
	--batch_size=512 \
	--gpus=${gpu} \
	--manual_seed=${manual_seed} \
	--experiment_root_path "./experiments_cram" \
	--exp_name=imagenet_resnet50_cram_plus_NM_rho${rho}_spgrad \
	--sam_rho=${rho}   \
	--wandb_group=imgnet_rn50_defaults_cram_plus_NM_sparse_grad \
	--wandb_name=CrAM_rho${rho} \
	--wandb_project "cram_imagenet_resnet50"


