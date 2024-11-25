import torch.nn as nn

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
        self.input_conv = nn.Conv1d(
            in_channels, hidden_dim, 7, padding="same", bias=False
        )
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
        self.output_conv = nn.Conv1d(
            hidden_dim // 2 ** (len(kernels)), 1, 7, padding="same", bias=False
        )
        self.tanh = nn.Tanh()

    def forward(self, spectrogram, **batch):
        x = self.input_conv(spectrogram)
        x = self.layers(x)
        predict = self.tanh(self.output_conv(self.relu(x)))
        return {"predict": predict[:, 0, :]}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
