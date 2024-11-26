import torch
import torch.nn as nn


class AdvGLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, msd_out_fake, mpd_out_fake, **batch):
        loss = 0.0
        for out_fake in [msd_out_fake, mpd_out_fake]:
            for out in out_fake:
                loss += ((out - 1) ** 2).mean()
        return loss


class AdvDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, msd_out_fake, msd_out_real, mpd_out_fake, mpd_out_real, **batch):
        loss = 0.0
        for out_fake_real in [
            [msd_out_fake, msd_out_real],
            [mpd_out_fake, mpd_out_real],
        ]:
            for fake, real in zip(out_fake_real[0], out_fake_real[1]):
                loss += ((real - 1) ** 2 + fake**2).mean()
        return loss
