"""VGG11/13/16/19 in Pytorch."""
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, color_channel=3, num_classes=10):
        super(VGG, self).__init__()
        self.color_channel = color_channel
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.color_channel
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def VGG11(color_channel=3, num_classes=10):
    return VGG('VGG11', color_channel=color_channel, num_classes=num_classes)


def VGG13(color_channel=3, num_classes=10):
    return VGG('VGG13', color_channel=color_channel, num_classes=num_classes)


def VGG16(color_channel=3, num_classes=10):
    return VGG('VGG16', color_channel=color_channel, num_classes=num_classes)


def VGG19(color_channel=3, num_classes=10):
    return VGG('VGG19', color_channel=color_channel, num_classes=num_classes)
