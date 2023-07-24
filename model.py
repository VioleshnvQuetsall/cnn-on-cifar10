import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as D


class CNN(nn.Module):
    def __init__(self, size=32, init_weights=False):
        super().__init__()

        def conv_layer(inc,
                       outc,
                       kernel_size=3,
                       padding=1,
                       batch_norm=True,
                       pool=False):
            nonlocal size
            aux = [
                nn.Conv2d(inc,
                          outc,
                          kernel_size=(kernel_size, kernel_size),
                          padding=padding)
            ]
            size += padding * 2 - (kernel_size - 1)
            if batch_norm:
                aux.append(nn.BatchNorm2d(outc))
            aux.append(nn.ReLU(inplace=True))

            if pool:
                aux.append(nn.MaxPool2d(kernel_size=2, stride=2))
                size //= 2

            return nn.Sequential(*aux)

        self.conv = nn.Sequential(
            conv_layer(3, 64, 7, 3, True, True),
            conv_layer(64, 128, 3, 1, True, True),
            conv_layer(128, 256, 3, 1, True, False),
            conv_layer(256, 256, 3, 1, True, True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(size * size * 256, 64),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(64, 10),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_weight_bias(self):
        weight, bias = [], []
        for name, p in self.named_parameters():
            if 'bias' in name:
                bias.append(p)
            elif 'weight' in name:
                weight.append(p)
        return weight, bias

    def frozen_batch_norm(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)