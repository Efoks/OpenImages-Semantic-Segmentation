import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    def __init__(self, channels_in, growth_rate, num_layers):
        super().__init__()

        self.layers = nn.ModuleList()
        current_channels = channels_in
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(current_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(current_channels, growth_rate, kernel_size=3, padding=1),
                nn.Dropout(0.5)
            ))
            current_channels += growth_rate


    def forward(self, x):
        featires = [x]
        for layer in self.layers:
            out = layer(torch.cat(featires, dim=1))
            featires.append(out)
        output = torch.cat(featires, dim=1)
        return output

class TransitionLayer(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()

        self.conv = nn.Sequential(
            nn.BatchNorm2d(channels_in),
            nn.Conv2d(channels_in, channels_out, kernel_size=1),
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, channels_in, block_config, groth_rate = 16):
        super().__init__()

        self.initial_processing = nn.Sequential(
            nn.Conv2d(channels_in, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.blocks = nn.ModuleList()

        channels = 64
        for num_layers in block_config:
            self.blocks.append(DenseBlock(channels, groth_rate, num_layers))
            channels += groth_rate * num_layers
            if num_layers != block_config[-1]:
                self.blocks.append(TransitionLayer(channels, channels // 2))
                channels = channels // 2

    def forward(self, x):
        x = self.initial_processing(x)
        for block in self.blocks:
            x = block(x)
        return x

class Bottleneck(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()

        self.atrouous_conv = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.atrouous_conv(x)

class DecoderBlock(nn.Module):
    def __init__(self, channels_in, channels_out, skip_channels):
        super().__init__()

        self.upconv = nn.ConvTranspose2d(channels_in, channels_out, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(channels_out + skip_channels, channels_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x, skip_x):
        x = self.upconv(x)
        x = torch.cat([x, skip_x], dim=1)
        return self.conv(x)

class Decoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, num_classes):
        super().__init__()

        self.blocks = nn.ModuleList()
        channels = encoder_channels[::-1]
        skip_channels = [0] + encoder_channels[::-1][1:]

        for channels_in, channels_skip, channels_out in zip(channels, skip_channels, decoder_channels):
            self.blocks.append(DecoderBlock(channels_in, channels_out, channels_skip))

        self.final_conv = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

    def forward(self, features):
        x = features[-1]
        for idx, block in enumerate(self.blocks):
            skip = features[-idx-2] if idx < len(features) - 1 else None
            x = block(x, skip) if skip is not None else block(x)
        return self.final_conv(x)

class FCN(nn.Module):
    def __init__(self, channels_in = 3, num_classes = 4):
        super().__init__()

        block_config = [2, 3, 4]
        encoder_channels = [64, 128, 256]
        decoder_channels = [128, 64, 32]
        self.encoder = Encoder(channels_in, block_config)
        self.bottleneck = Bottleneck(64, 64)
        self.decoder = Decoder(encoder_channels, decoder_channels, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        x = self.bottleneck(features)
        x = self.decoder([features, x])
        return x



if __name__ == '__main__':
    test_encoder_output_shape()
    test_bottleneck_functionality()
    test_decoder_integration()
    test_fcn_model_integration()
