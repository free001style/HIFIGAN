import torch
import torch.nn as nn


class MelSpecLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, spectrogram, spectrogram_predict, **batch):
        """
        Args:
            spectrogram (Tensor): Spectrogram of original sample.
            spectrogram_predict (Tensor): Spectrogram of generated sample.
        """
        return nn.functional.l1_loss(spectrogram, spectrogram_predict)
