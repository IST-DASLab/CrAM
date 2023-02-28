# Code for CrAM: A Compression-Aware Minimizer

## About

This repository contains the implementation of the CrAM optimizer, proposed in the ICLR 2023 accepted paper **CrAM: A Compression-Aware Minimizer**.
Here you can find scripts required for reproducing the results on CIFAR10 and ImageNet, as presented in the CrAM paper.

## Structure

- `main.py` is the main script for launching a training run using CrAM or a different optimizer.
 In addition to passing the arguments related to dataset and architecture, the training details (type of optimizer, hyperparameters,
  learning rate scheduler) are passed through a config file. Examples of config files for CrAM training can be found in the `configs/` folder
- `manager.py` contains the Manager class, which trains the model with the desired configs, through the `run` method
- `optimization/` module contains the custom optimizers (Topk-CrAM and SAM), and a custom cosine annealing learning rate scheduler 
with linear rate warm-up; we also include Topk-CrAM for N:M sparsity patterns
- `load_training_configs.py` contains functions for loading the optimizer, learning rate scheduler and training hyperparameters 
from the config file
- `models/` contains the available types of models; additionally, we use the ResNet ImageNet models as defined in the Torchvision library
- `utils/` contains different utilities scripts (e.g. for loading datasets, saving and loading checkpoints, 
functions for one-shot pruning or fixing batch norm statistics)
- `generate_calibration_dset.py` creates a calibration set of 1000 Imagenet training samples, and copies them to a different folder; 
to be used when doing Batch Norm tuning after pruning
- `demo_get_one_shot_pruning_results.py` script that loads a trained CrAM checkpoint, prunes it one-shot to a desired sparsity, 
does Batch Norm tuning on a calibration set, 
and checks the validation accuracy after each of these operations


## How to run

We provide a few sample bash scripts in the `run_scripts` folder, to reproduce our results using 
CrAM+ Multi (trained with sparse intermediate gradients) on CIFAR10 and ImageNet, 
as well as CrAM+ k95 on ResNet18/VGG16 (CIFAR10).

Sample bash scripts:
- CIFAR10: `run_cifar10_resnet20_cram_demo.sh`, 
`run_cifar10_big_resnet18_cram_plus_k95_sparse_grad.sh`, `run_cifar10_vgg16_cram_plus_k95_sparse_grad.sh` (for RN20/ RN18/ VGG16)
- ImageNet: `run_imagenet_resnet50_cram_plus_multi_demo.sh`, 
`run_imagenet_resnet50_cram_plus_NM_demo.sh` (for RN50)

`bash RUN_SCRIPT {GPU id}` (GPU id only needed for CIFAR10 scripts)

To obtain the results after one-shot pruninig + BNT:
- change the paths to the dataset and provide path to trained checkpoint, and run: 
`python demo_get_one_shot_pruning_results.py --use_calib --calib_size 1000 --sparsity 0.9`
- for BNT on fixed calibration set (only on ImageNet), first create the calibration set using `generate_calibration_dset.py` and then
`python demo_get_one_shot_pruning_results.py --fixed_calib --use_calib --calib_size 1`
- the default pruning distribution is `global`, but `uniform` and `N:M` are also available, 
by changing the `--pruning` argument appropriately. 

We also use Weights & Biases (Wandb) for tracking our experiments. This can be enabled through `--use_wandb` inside the bash scripts. Not enabling it will use the print function by default.


## Integrating the CrAM Optimizer

Note that the CrAM optimizer can be easily integrated inside any code base. In what follows, 
we provide an example of an integration.

Define the optimizer and learning rate scheduler. Use `plus_version=True` and  `sparse_grad=True`
for best results:

```python
base_optimizer = torch.optim.SGD
optimizer = TopkCrAM(model.parameters(), base_optimizer, rho=0.05, sparsities=[0.5, 0.7, 0.9], 
                     plus_version=True, sparse_grad=True,
                     lr=0.1, momentum=0.9, weight_decay=0.0001)
lr_scheduler = CosineLR(optimizer, warmup_length=5, end_epoch=100)
```


Add the following inside the training loop:
```python
model.train()

for data, target in train_loader:
    def closure():
        closure_outputs = model(data)
        closure_loss = criterion(closure_outputs, target)
        closure_loss.backward()
        return closure_loss
        
    optimizer.zero_grad()
    
    outputs = model(data)
    loss = criterion(outputs, target)
    loss.backward() 
    
    disable_running_stats(model)  # ensures that BN statistics are computed only on the unperturbed model   
    optimizer.step(closure)  # does the CrAM update under the hood
    enable_running_stats(model)  # enable back the the BN statistics for the next batch
    
lr_scheduler.step()
```


## CrAM-trained checkpoints

 We also provide ImageNet/ResNet50 models trained with CrAM:

- [CrAM+Multi](https://seafile.ist.ac.at/f/0eacc1ffa37248f5b984/?dl=1) 
trained using {50%, 70%, 90%} sparsity levels, with 77.3%  ImageNet validation accuracy. This model 
can achieve 75.8% accuracy at 80% sparsity and 74.8% accuracy at 90% sparsity, after Batch Norm tuning.
- [CrAM+ N:M](https://seafile.ist.ac.at/f/042504b22321432bbb4d/?dl=1) 
trained using 2:4 and 4:8 sparsity, with 77.3% ImageNet validation accuracy. This model can achieve
77.2% accuracy for 2:4 sparsity and 77.2% accuracy for 4:8 sparsity, after Batch Norm tuning.

## Requirements

- python 3.9
- torch 1.8.1 
- torchvision 0.9.1
- wandb 0.12.17

The experiments are performed using PyTorch1.8.1, but the code has also been tested with newer 
PyTorch versions (e.g. 1.12), and obtained similar results.


## Acknowledgements


Our work is inspired from the Sharpness-Aware Minimization (SAM) work (Foret et al., ICLR 2021).

We would also like to thank David Samuel for providing a PyTorch implementation of the SAM optimizer, 
which the implementation of CrAM is inspired from. The PyTorch SAM implementation is available here:
`https://github.com/davda54/sam`.



## BibTex

If you found this repository useful, please consider citing our work:
```
@article{peste2023cram,
  title={CrAM: A Compression-Aware Minimizer},
  author={Peste, Alexandra and Vladu, Adrian and Kurtic, Eldar and Lampert, Christoph and Alistarh, Dan},
  journal={International Conference on Learning Representations (ICLR)},
  year={2023}
}
```
