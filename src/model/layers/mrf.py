import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation):
        super().__init__()
        self.layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(
                        in_channels,
                        in_channels,
                        kernel_size,
                        dilation=dilation[i],
                        padding="same",
                    ),
                )
                for i in range(len(dilation))
            ]
        )

    def forward(self, x):
        residual = x
        x = self.layers(x)
        return x + residual


class MRFBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation):
        super().__init__()
        # print(len(dilation))
        self.layers = nn.Sequential(
            *[
                ResBlock(in_channels, kernel_size, dilation[i])
                for i in range(len(dilation))
            ]
        )

    def forward(self, x):
        return self.layers(x)


class MRF(nn.Module):
    def __init__(self, in_channels, kernels, dilation):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MRFBlock(in_channels, kernels[i], dilation[i])
                for i in range(len(kernels))
            ]
        )

    def forward(self, x):
        out = torch.zeros_like(x)
        for layer in self.layers:
            out += layer(x)
        return out
