import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

from src.model.layers.conv import Conv


class SubMSD(nn.Module):
    def __init__(self, use_weight_norm=True):
        super().__init__()
        norm = weight_norm if use_weight_norm else spectral_norm

        self.layers = nn.ModuleList(
            [
                Conv(
                    1,
                    16,
                    15,
                    padding=7,
                    activation=nn.LeakyReLU(0.1),
                    normalization=norm,
                    is_2d=False,
                ),
                Conv(
                    16,
                    64,
                    41,
                    4,
                    groups=4,
                    padding=20,
                    activation=nn.LeakyReLU(0.1),
                    normalization=norm,
                    is_2d=False,
                ),
                Conv(
                    64,
                    256,
                    41,
                    4,
                    groups=16,
                    padding=20,
                    activation=nn.LeakyReLU(0.1),
                    normalization=norm,
                    is_2d=False,
                ),
                Conv(
                    256,
                    1024,
                    41,
                    4,
                    groups=64,
                    padding=20,
                    activation=nn.LeakyReLU(0.1),
                    normalization=norm,
                    is_2d=False,
                ),
                Conv(
                    1024,
                    1024,
                    41,
                    4,
                    groups=256,
                    padding=20,
                    activation=nn.LeakyReLU(0.1),
                    normalization=norm,
                    is_2d=False,
                ),
                Conv(
                    1024,
                    1024,
                    5,
                    padding=2,
                    activation=nn.LeakyReLU(0.1),
                    normalization=norm,
                    is_2d=False,
                ),
            ]
        )
        self.output_conv = Conv(1024, 1, 3, padding=1, normalization=norm, is_2d=False)

    def forward(self, predict):
        x = predict.unsqueeze(1)
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        return self.output_conv(x), feats


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = nn.ModuleList(
            [
                SubMSD(False),
                SubMSD(),
                SubMSD(),
            ]
        )

    def forward(self, predict, audio, **batch):
        out_fake, feats_fake, out_real, feats_real = [], [], [], []
        for disc in self.discriminator:
            out, feats = disc(predict)
            out_fake.append(out)
            feats_fake.append(feats)
            predict = nn.functional.adaptive_avg_pool1d(predict, predict.shape[1] // 2)

            out, feats = disc(audio)
            out_real.append(out)
            feats_real.append(feats)
            audio = nn.functional.adaptive_avg_pool1d(audio, audio.shape[1] // 2)
        return out_fake, feats_fake, out_real, feats_real
