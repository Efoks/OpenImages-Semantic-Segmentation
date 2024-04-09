from torchvision.models.segmentation import fcn_resnet50
import torch
import torchvision.transforms.functional
from torch import nn
import torch.nn.functional as F

def resnet_model():
    return fcn_resnet50(pretrained=False, num_classes=4)


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


class UpConv2x2(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()

        self.up = nn.ConvTranspose2d(channels_in, channels_out, kernel_size = 2, stride = 2)

    def forward(self, x):
        return self.up(x)


class CopyAndCrop(nn.Module):
    def forward(self, x, contracting_x):
        contracting_x = torchvision.transforms.functional.center_crop(contracting_x, [x.shape[2], x.shape[3]])
        x = torch.cat([x, contracting_x], dim = 1)
        return x


class UNet(nn.Module):

    def __init__(self, channels_in, channels_out):
        super().__init__()

        self.down_convulation = nn.ModuleList([DoubleConvolution(i, j)
                                               for i, j in [(channels_in, 64), (64, 128), (128, 256), (256, 512)]])

        self.max_pool = nn.ModuleList([MaxPool2x2() for _ in range(4)])

        self.middle_convulation = DoubleConvolution(512, 1024)

        self.up_sample = nn.ModuleList([UpConv2x2(i, j)
                                      for i, j in [(1024, 512), (512, 256), (256, 128), (128, 64)]])

        self.up_convulation = nn.ModuleList([DoubleConvolution(i, j)
                                             for i, j in [(1024, 512), (512, 256), (256, 128), (128, 64)]])

        self.concat = nn.ModuleList([CopyAndCrop() for _ in range(4)])

        self.last_convulation = nn.Conv2d(64, channels_out, kernel_size = 1)

    def forward(self, x):
        passed = []

        for i in range(len(self.down_convulation)):
            x = self.down_convulation[i](x)
            passed.append(x)
            x = self.max_pool[i](x)

        return x

def unet_model():
    return UNet(3, 4)


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

def test_aspp():
    aspp = ASPP(3, 4)
    x = torch.rand(1, 3, 224, 224)
    y = aspp(x)
    assert y.shape == (1, 4, 224, 224)

def real_test_aspp():
    import os
    from PIL import Image
    import numpy as np
    import config as cfg
    import matplotlib.pyplot as plt

    aspp = ASPP(3, 4)
    images = os.listdir(os.path.join(cfg.DATA_DIR, 'dog', 'images'))
    image_path = os.path.join(cfg.DATA_DIR, 'dog', 'images', images[0])
    image = Image.open(image_path).resize((256, 256))
    image_aspp = torchvision.transforms.functional.to_tensor(image)
    image_aspp = image_aspp.unsqueeze(0)
    y = aspp(image_aspp)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image)
    ax[1].imshow(np.argmax(y[0].detach().numpy(), axis=0))
    plt.show()

def real_test_fcn():
    import os
    from PIL import Image
    import numpy as np
    import config as cfg
    import matplotlib.pyplot as plt

    fcn = FCN(3, 4)
    images = os.listdir(os.path.join(cfg.DATA_DIR, 'dog', 'images'))
    image_path = os.path.join(cfg.DATA_DIR, 'dog', 'images', images[0])
    image = Image.open(image_path).resize((256, 256))
    image_fcn = torchvision.transforms.functional.to_tensor(image)
    image_fcn = image_fcn.unsqueeze(0)
    y = fcn(image_fcn)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image)
    ax[1].imshow(np.argmax(y[0].detach().numpy(), axis=0))
    plt.show()

if __name__ == "__main__":
    real_test_fcn()