import torch
import torch.nn as nn


class AdvGLoss(nn.Module):
    """
    Adversarial loss for generator.
    """

    def __init__(self):
        super().__init__()

    def forward(self, msd_out_fake, mpd_out_fake, **batch):
        """
        Args:
            msd_out_fake (list[Tensor]): List of MSD outputs for fake samples.
            mpd_out_fake (list[Tensor]): List of MPD outputs for fake samples.
        """
        loss = 0.0
        for out_fake in [msd_out_fake, mpd_out_fake]:
            for out in out_fake:
                loss += ((out - 1) ** 2).mean()
        return loss


class AdvDLoss(nn.Module):
    """
    Adversarial loss for discriminators.
    """

    def __init__(self):
        super().__init__()

    def forward(self, msd_out_fake, msd_out_real, mpd_out_fake, mpd_out_real, **batch):
        """
        Args:
            msd_out_fake (list[Tensor]): List of MSD outputs for fake samples.
            msd_out_real (list[Tensor]): List of MSD outputs for real samples.
            mpd_out_fake (list[Tensor]): List of MPD outputs for fake samples.
            mpd_out_real (list[Tensor]): List of MPD outputs for real samples.
        """
        loss = 0.0
        for out_fake_real in [
            [msd_out_fake, msd_out_real],
            [mpd_out_fake, mpd_out_real],
        ]:
            for fake, real in zip(out_fake_real[0], out_fake_real[1]):
                loss += ((real - 1) ** 2 + fake**2).mean()
        return loss
