import torch

from models.resnet_cifar10 import *
from models.big_resnet_cifar10 import *
from models.vgg_cifar10 import *
from models.mobilenet import *

from torchvision.models import resnet18, resnet34, resnet50 


CIFAR10_MODELS = ['resnet20', 'resnet32', 'resnet18_cifar_big', 'VGG16']
CIFAR100_MODELS = ['resnet20', 'resnet18_big_cifar']
IMAGENET_MODELS = ['resnet18', 'resnet34', 'resnet50', 'mobilenet']


def get_model(name, dataset, pretrained=False):
    if (name.startswith('resnet') or name.startswith('VGG')) and (dataset == 'cifar10'):
        return globals()[name]()
    elif (name.startswith('resnet') or name.startswith('VGG')) and (dataset == 'cifar100'):
        return globals()[name](num_classes=100)
    elif 'mobilenet' in name:
        return globals()[name]()
    elif ('resnet' in name) and (dataset == 'imagenet'):
        return globals()[name](pretrained=pretrained)
    else:
        raise NotImplementedError('Not a supported (model, dataset) pair')




     


