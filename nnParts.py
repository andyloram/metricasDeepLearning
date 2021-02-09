import torch
import torch.nn as nn


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, ks=3, max_pool=None):
        super().__init__()
        self.double_conv = nn.Sequential(
            SingleConv(in_channels, out_channels, ks),
            SingleConv(out_channels, out_channels, ks, max_pool),
        )

    def forward(self, x):
        return self.double_conv(x)


class SingleConv(nn.Module):

    def __init__(self, in_channels, out_channels, ks=3, max_pool=None):
        super().__init__()
        if max_pool:
            self.single_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, ks),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(max_pool)
            )
        else:
            self.single_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, ks),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.single_conv(x)
