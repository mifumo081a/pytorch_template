import torch
import torch.nn as nn


class DepthwiseSepConv(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, padding_mode="zeros", bias=True):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, stride=stride, padding_mode=padding_mode, groups=in_channels, bias=bias),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, padding_mode="zeros", bias=True,
                bn=False, gelu=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, bias=bias)
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        if gelu:
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x
    

class DeConvBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, bias=True):
        super().__init__()
        self.tconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        out = self.tconv(x)
        return out