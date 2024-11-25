import torch.nn as nn

from src.model.layers.conv import Conv
from src.model.layers.mrf import MRF


class GeneratorBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, mrf_kernels, mrf_dilation
    ):
        super().__init__()
        self.layer = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size,
                kernel_size // 2,
                padding=(kernel_size - kernel_size // 2) // 2,
            ),
            MRF(out_channels, mrf_kernels, mrf_dilation),
        )

    def forward(self, x):
        return self.layer(x)


class Generator(nn.Module):
    def __init__(
        self,
        in_channels=80,
        hidden_dim=128,
        kernels=[16, 16, 4, 4],
        mrf_kernels=[3, 7, 11],
        mrf_dilation=[[[1, 1], [3, 1], [5, 1]]] * 3,
    ):
        super().__init__()
        self.input_conv = Conv(in_channels, hidden_dim, 7, padding="same", is_2d=False)
        self.layers = nn.Sequential(
            *[
                GeneratorBlock(
                    hidden_dim // 2**i,
                    hidden_dim // 2 ** (i + 1),
                    kernels[i],
                    mrf_kernels,
                    mrf_dilation,
                )
                for i in range(len(kernels))
            ]
        )
        self.relu = nn.LeakyReLU(0.1)
        self.output_conv = Conv(
            hidden_dim // 2 ** (len(kernels)),
            1,
            7,
            padding="same",
            activation=nn.Tanh(),
            is_2d=False,
        )

    def forward(self, spectrogram, **batch):
        x = self.input_conv(spectrogram)
        x = self.layers(x)
        predict = self.output_conv(self.relu(x))
        return {"predict": predict[:, 0, :]}
