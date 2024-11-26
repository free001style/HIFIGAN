import torch
from torch import nn

from src.loss.losses.adversarial_loss import AdvDLoss, AdvGLoss
from src.loss.losses.feats_matching_loss import FeatsMatchingLoss
from src.loss.losses.melspec_loss import MelSpecLoss


class GeneratorLoss(nn.Module):
    def __init__(self, mel_lambda=45, fm_lambda=2):
        super().__init__()
        self.adv_loss = AdvGLoss()
        self.mel_loss = MelSpecLoss()
        self.fm_loss = FeatsMatchingLoss()
        self.mel_lambda = mel_lambda
        self.fm_lambda = fm_lambda

    def forward(self, **batch):
        adv_g_loss = self.adv_loss(**batch)
        mel_loss = self.mel_loss(**batch)
        fm_loss = self.fm_loss(**batch)
        loss = adv_g_loss + self.mel_lambda * mel_loss + self.fm_lambda * fm_loss
        return {
            "g_loss": loss,
            "adv_g_loss": adv_g_loss,
            "mel_loss": mel_loss,
            "fm_loss": fm_loss,
        }


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.adv_loss = AdvDLoss()

    def forward(self, **batch):
        return {"d_loss": self.adv_loss(**batch)}
