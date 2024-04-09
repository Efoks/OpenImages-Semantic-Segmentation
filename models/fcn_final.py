"""
Author: Edvardas Timoscenka
LSP number:
Model Version: v0.1

The full model implementation with training, dataloaders, etc. can be found in the training_v2.py file in the github repository
GitHub: .
"""

import torch
from torch import nn


class DenseBlock(nn.Module):
    """
    Based on: https://d2l.ai/chapter_convolutional-modern/densenet.html
    """
    def __init__(self, channels_in, growth_rate, num_layers):
        """
        Initialize a DenseBlock.
        Added Dropout with 0.3, because the model otherwise overfits the data.

        Args:
            channels_in (int): The number of input channels.
            growth_rate (int): The growth rate controls how many filters to add each layer (k in the paper).
            num_layers (int): The number of layers in each dense block.
        """
        super().__init__()

        self.layers = nn.ModuleList()
        current_channels = channels_in
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(current_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(current_channels, growth_rate, kernel_size=3, padding=1),
                nn.Dropout(0.3)
            ))
            current_channels = current_channels + growth_rate

    def forward(self, x):
        """
        Forward pass through the DenseBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        for layer in self.layers:
            y = layer(x)
            x = torch.cat((x, y), dim=1)
        return x


class TransitionBlock(nn.Module):
    """
    Based on: https://d2l.ai/chapter_convolutional-modern/densenet.html
    """
    def __init__(self, channels_in, channels_out):
        """
        Initialize a TransitionBlock.
        Used in combination with the DenseBlock to half the channels,
        thus decreasing the complexity.

        Args:
            channels_in (int): The number of input channels.
            channels_out (int): The number of output channels.
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.BatchNorm2d(channels_in),
            nn.Conv2d(channels_in, channels_out, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        """
        Forward pass through the TransitionBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.net(x)
        return x


class Encoder(nn.Module):
    """
    Encoder consisting of DenseBlock that is going to be used in the model.
    """
    def __init__(self, channels_in, channels_out, block_config):
        """
        Initialize an Encoder.
        Block config defines the number of dense blocks.
        Growth rate is set as 32.

        Args:
            channels_in (int): The number of input channels.
            channels_out (int): The number of output channels.
            block_config (list): The number of layers in each dense block.
        """
        super().__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.blocks = nn.ModuleList()

        channels = channels_out
        for idx, num_layers in enumerate(block_config):
            self.blocks.append(DenseBlock(channels, 32, num_layers))
            channels = channels + 32 * num_layers

            # Transition is applied only in between the DenseBlock
            if idx != len(block_config) - 1:
                self.blocks.append(TransitionBlock(channels, channels // 2))
                channels = channels // 2

    def forward(self, x):
        """
        Forward pass through the Encoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.initial_conv(x)
        for block in self.blocks:
            x = block(x)
        return x


class Bottleneck(nn.Module):
    """
    Used in between encode and decoder. As a bottleneck simple convulation is used.
    """
    def __init__(self, channels_in, channels_out):
        """
        Initialize a Bottleneck.

        Args:
            channels_in (int): The number of input channels.
            channels_out (int): The number of output channels.
        """
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass through the Bottleneck.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv(x)
        return x

class Decoder(nn.Module):
    """
    Decoder is based on atreous convulations.
    """
    def __init__(self, channels_in, channels_out, num_layers, dilation_rate):
        """
        Initialize a Decoder.

        Args:
            channels_in (int): The number of input channels.
            channels_out (int): The number of output channels.
            num_layers (int): The number of layers in each dense block.
            dilation_rate (list): The dilation rate for each layer.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.dilation_rate = dilation_rate

        for i in range(num_layers):
            padding = dilation_rate[i] * (3 - 1) // 2
            self.layers.append(nn.Sequential(
                nn.Conv2d(channels_in if i == 0 else channels_out, channels_out, kernel_size=3, padding=padding, dilation=dilation_rate[i]),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            ))

        self.final_conv = nn.Conv2d(channels_out, channels_out, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through the Decoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        for layer in self.layers:
            x = layer(x)

        x = self.final_conv(x)
        return x

class NeuralNet(nn.Module):
    """
    The main model who combines all preceding ones into unified structure.
    """
    def __init__ (self, channels_in, num_classes, block_config, num_layers, dilation_rate):
        """
        Initialize a NeuralNet.

        Args:
            channels_in (int): The number of input channels.
            num_classes (int): The number of output classes.
            block_config (list): The number of layers in each dense block.
            num_layers (int): The number of layers in each dense block.
            dilation_rate (list): The dilation rate for each layer.
        """
        super().__init__()

        self.encoder = Encoder(channels_in, 64, block_config)

        self.bottleneck = Bottleneck(208, 208)

        self.decoder = Decoder(208, 256, num_layers, dilation_rate)

        self.final_conv = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through the NeuralNet.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        x = self.final_conv(x)
        return x

def fcn_model():
    return NeuralNet(3, 4, [2, 3, 4], 4, [1, 2, 4, 8])

# ------------------------------------------
#       Code down below is used for testing
#-------------------------------------------

# if __name__ == '__main__':
#     encoder = Encoder(3, 64, [2, 3, 4])
#     bottleneck = Bottleneck(208, 208)
#     model = Decoder(208, 256, 4, [1, 2, 4, 8])
#
#     x = torch.randn((1, 3, 256, 256))
#
#     encoder_output = encoder(x)
#     print(encoder_output.shape)
#     bottleneck_output = bottleneck(encoder_output)
#     print(bottleneck_output.shape)
#     model_output = model(bottleneck_output)
#     print(model_output.shape)
#     final_conv = nn.Conv2d(256, 4, kernel_size=1)
#     print(final_conv(model_output).shape)
#
#     model = NeuralNet(3, 4, [2, 3, 4], 4, [1, 2, 4, 8])
#     output = model(x)
#     print(output.shape)