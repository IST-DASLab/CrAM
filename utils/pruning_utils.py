import torch
import math
import torch.nn as nn
from utils.utils import percentile


def one_shot_prune(model, sparsity):
    masks_dict = {}
    all_weights_stats = None
    for name, par in model.named_parameters():
        if len(par.shape)>1:
            weight_stat = par.data.abs().view(-1)
            if all_weights_stats is None:
                all_weights_stats = weight_stat
            else:
                all_weights_stats = torch.cat((all_weights_stats, weight_stat))
    threshold = percentile(all_weights_stats, sparsity)
    for name, par in model.named_parameters():
        if len(par.shape) > 1:
            masks_dict[name] = (par.data.abs() > threshold).float()
            par.data *= masks_dict[name]
    return masks_dict


def one_shot_uniform_prune(model, sparsity):
    masks_dict = {}
    all_params = list(model.named_parameters())
    for i in range(1, len(all_params)-2):
        param = all_params[i][1]
        if len(param.shape) < 2:
            continue
        param_stats = torch.abs(param.data).view(-1)
        k = math.ceil(param_stats.numel() * (1. - sparsity))
        threshold = torch.topk(param_stats, k)[0][-1]
        mask = (torch.abs(param.data) > threshold).float()
        param.data *= mask
        masks_dict[all_params[i][0]] = mask
    return masks_dict


# code adapted from: https://github.com/NM-sparsity/NM-sparsity/blob/main/devkit/sparse_ops/sparse_ops.py
def one_shot_NM_prune(model, N, M, skip_last=False):
    masks_dict = {}
    for name, module in model.named_modules():
        mask = None
        if isinstance(module, nn.Conv2d):
            length = module.weight.data.numel()
            group = int(length / M)
            weight_temp = module.weight.data.abs().permute(0, 2, 3, 1).reshape(group, M)
            index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]

            mask = torch.ones(weight_temp.shape, device=weight_temp.device)
            mask = mask.scatter_(dim=1, index=index, value=0).reshape(module.weight.permute(0, 2, 3, 1).shape)
            mask = mask.permute(0, 3, 1, 2)

        if (not skip_last) and isinstance(module, nn.Linear):
            length = module.weight.data.numel()
            group = int(length / M)
            weight_temp = module.weight.data.abs().reshape(group, M)
            index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]

            mask = torch.ones(weight_temp.shape, device=weight_temp.device)
            mask = mask.scatter_(dim=1, index=index, value=0).reshape(module.weight.shape)

        if mask is not None:
            masks_dict[name] = mask
            module.weight.data *= masks_dict[name]
    
    return masks_dict


def apply_masks(model, masks_dict):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            module.weight.data *= masks_dict[name]


def test_sparsity(model):
    total_zeros = 0.
    total_params = 0.
    for param in model.parameters():
        total_zeros += (param.data == 0.).float().sum().item()
        total_params += param.data.numel()
    sparsity = total_zeros / total_params
    return sparsity

