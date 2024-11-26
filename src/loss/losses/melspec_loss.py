import torch
import torch.nn as nn


class MelSpecLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, spectrogram, spectrogram_fake, **batch):
        return nn.functional.l1_loss(spectrogram, spectrogram_fake)
