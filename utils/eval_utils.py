import torch
import torch.nn as nn
import torch.nn.functional as F


def eval_model(model, loader, device):
    model.eval()
    eval_loss = 0
    correct = 0
    with torch.no_grad():
        for in_tensor, target in loader:
            in_tensor, target = in_tensor.to(device), target.to(device)
            output = model(in_tensor)
            eval_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    eval_loss = eval_loss / len(loader.dataset)
    eval_acc = correct / len(loader.dataset)
    return eval_loss, eval_acc


def fix_bn_stats(model, loader, device, iters=100):
    i = 0
    print("train loader size:", len(loader.dataset)) 
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = 0.1

    model.train()
    with torch.no_grad():    
        while i < iters:
            for in_tensor, target in loader:
                if i == iters:
                    print("END BNT:", i)
                    break
                model(in_tensor.to(device))
                i += 1
    model.eval()
    return model

