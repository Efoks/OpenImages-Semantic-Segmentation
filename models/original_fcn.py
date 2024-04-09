import torch
import torchvision.transforms.functional
from torch import nn
import torch.nn.functional as F


class DoubleConvolution(nn.Module):
    def __init__ (self, channels_in, channels_out):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size = 3, padding = 1),
            nn.ReLU(inplace= True),
            nn.Conv2d(channels_out, channels_out, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        return self.double_conv(x)


class MaxPool2x2(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(x)

class ASPP(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(ASPP, self).__init__()

        self.dilation_r = [1, 6, 12, 18]
        self.aspp = nn.ModuleList([nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, padding=dilation, dilation=dilation),
            nn.ReLU(inplace=True)
            ) for dilation in self.dilation_r])
        self.final_conv = nn.Conv2d(len(self.dilation_r) * channels_out, channels_out, 1)

    def forward(self, x):
        aspp_out = [aspp(x) for aspp in self.aspp]
        aspp_out = torch.cat(aspp_out, dim=1)
        return self.final_conv(aspp_out)

class FCN(nn.Module):
    def __init__(self, channels_in, channels_out, dropout_rate = 0.5):
        super(FCN, self).__init__()

        self.convulations = nn.ModuleList([DoubleConvolution(i, j) for i, j in [(channels_in, 64), (64, 128), (128, 256), (256, 512)]])
        self.dropout = nn.Dropout(dropout_rate)
        self.max_pool = nn.ModuleList([MaxPool2x2() for _ in range(4)])
        self.aspp = ASPP(512, 256)

        self.skip_conv = nn.Conv2d(64, 256, 1)

        self.last_convulation = nn.Conv2d(256, channels_out, kernel_size = 1)

    def forward(self, x):
        passed = []

        for i in range(len(self.convulations)):
            x = self.convulations[i](x)
            x = self.dropout(x)
            passed.append(x)
            x = self.max_pool[i](x)

        x = self.aspp(x)
        skip_con = self.skip_conv(passed[0])
        x += F.interpolate(skip_con, x.size()[2:], mode='bilinear', align_corners=False)
        x = self.last_convulation(x)
        return F.interpolate(x, scale_factor=16, mode='bilinear', align_corners=False)

def fcn_model():
    return FCN(3, 4)

