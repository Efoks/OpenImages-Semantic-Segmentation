import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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


class TransitionBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels_in)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        return x


class MaxPool2x2(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(x)


class DoubleConvolution(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class ASPP(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(ASPP, self).__init__()

        self.dilation_r = [1, 6, 18]

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
    def __init__(self, channels_in, channels_out):
        super().__init__()

        self.encoders = nn.ModuleList([
            DenseBlock(channels_in, growth_rate=32, num_layers=4),
            TransitionBlock(channels_in + 4 * 32, (channels_in + 4 * 32) // 2),
            DenseBlock((channels_in + 4 * 32) // 2, growth_rate=32, num_layers=4),
            TransitionBlock(193, 193 // 2),
            MaxPool2x2(),
        ])
        self.aspp = ASPP(96, 96)

        self.decoders = nn.ModuleList([
            nn.ConvTranspose2d(channels_in + 10 * 32 // 4, 128, 2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            DoubleConvolution(128 + channels_in + 8 * 32 // 2, 128),
            nn.ConvTranspose2d(128, 64, 2, 2)
        ])

        self.skip_conv = nn.Sequential(
            nn.Conv2d(channels_in + 4 * 32 // 2, 64, 1)
        )

        self.last_convulation = nn.Conv2d(64, channels_out, 1)

    def forward(self, x):
        encoder_outputs = []
        for block in self.encoders:
            x = block(x)
            print(f'Block: {block}, x: {x.shape}')
            encoder_outputs.append(x)

        x = self.aspp(x)

        for block, endocer_output in zip(self.decoders, encoder_outputs[::-1]):
            x = block(x)
            if isinstance(block, DoubleConvolution):
                skip = self.skip_conv(endocer_output)
                print(x.shape, skip.shape)
                x = torch.cat([x, skip], dim=1)
            print(f'Block: {block}, x: {x.shape}')

        x = self.last_convulation(x)
        return x


def train_loop(model, data_loader, optimizer, criterion):

    model.train()
    for batch_idx, data in enumerate(data_loader):
        output = model(data)
        dummy_target = torch.zeros_like(output)

        optimizer.zero_grad()
        loss = criterion(output, dummy_target)

        loss.backward()
        optimizer.step()

def main():
    channels_in = 3
    growth_rate = 32
    num_layers = 4
    model = FCN(channels_in, 4)

    data_loader = torch.utils.data.DataLoader(
        torch.randn(20, 3, 256, 256),
        batch_size=4
    )

    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    train_loop(model, data_loader, optimizer, criterion)

if __name__ == "__main__":
    main()

