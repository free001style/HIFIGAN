import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from src.model.layers.conv import Conv


class SubMPD(nn.Module):
    """
    One of the Multi Period Discriminators
    """

    def __init__(self, p):
        """
        Args:
            p (int): The period for input audio.
        """
        super().__init__()
        self.p = p
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    Conv(
                        1 if i == 0 else 2 ** (5 + i),
                        2 ** (6 + i),
                        (5, 1),
                        (3, 1),
                        (2, 0),
                        activation=nn.LeakyReLU(0.1),
                        normalization=weight_norm,
                    )
                )
                for i in range(4)
            ]
        )
        self.output_conv = nn.ModuleList(
            [
                nn.Sequential(
                    Conv(
                        2**9,
                        1024,
                        (5, 1),
                        padding="same",
                        activation=nn.LeakyReLU(0.1),
                        normalization=weight_norm,
                    )
                ),
                Conv(1024, 1, (3, 1), padding="same", normalization=weight_norm),
            ]
        )

    def forward(self, predict):
        """
        Args:
            predict (Tensor): [B, T] audio tensor.
        Returns:
            output (Tensor): Output tensor of discriminator.
            feats (list[Tensor]): Feature maps from all layers.
        """
        x = predict.unsqueeze(1)
        b, c, t = x.shape
        if t % self.p:
            x = nn.functional.pad(x, (0, self.p - t % self.p), "constant", 0)
        x = x.view(b, c, x.shape[-1] // self.p, self.p)
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        x = self.output_conv[0](x)
        feats.append(x)
        return self.output_conv[-1](x), feats


class MultiPeriodDiscriminator(nn.Module):
    """
    Multi Period Discriminator
    """

    def __init__(self, p=[2, 3, 5, 7, 11]):
        """
        Args:
            p (list[int]): List of periods for SubMPD.
        """
        super().__init__()
        self.discriminator = nn.ModuleList([SubMPD(p_) for p_ in p])

    def forward(self, predict, audio, **batch):
        """
        Args:
            predict (Tensor): [B, T] generated audio.
            audio (Tensor): [B, T] ground truth audio.
        Return:
            mpd_out_fake (list[Tensor]): List of outputs from SubMPD for fake sample.
            mpd_out_real (list[Tensor]): List of outputs from SubMPD for real sample.
            mpd_feats_fake (list[list[Tensor]]): List of features from SubMPD layer for fake sample.
            mpd_feats_real (list[list[Tensor]]): List of features from SubMPD layer for real sample.
        """
        out_fake, feats_fake, out_real, feats_real = [], [], [], []
        for disc in self.discriminator:
            out, feats = disc(predict)
            out_fake.append(out)
            feats_fake.append(feats)

            out, feats = disc(audio)
            out_real.append(out)
            feats_real.append(feats)
        return {
            "mpd_out_fake": out_fake,
            "mpd_feats_fake": feats_fake,
            "mpd_out_real": out_real,
            "mpd_feats_real": feats_real,
        }
