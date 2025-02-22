import torch
import torch.nn as nn


class FeatsMatchingLoss(nn.Module):
    """
    Feature Matching Loss
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, mpd_feats_fake, mpd_feats_real, msd_feats_fake, msd_feats_real, **batch
    ):
        """
        Args:
            mpd_feats_fake (list[list[Tensor]]): List of MPD features from all layers for fake samples.
            mpd_feats_real (list[list[Tensor]]): List of MPD features from all layers for real samples.
            msd_feats_fake (list[list[Tensor]]): List of MSD features from all layers for fake samples.
            msd_feats_real (list[list[Tensor]]): List of MSD features from all layers for real samples.
        """
        loss = 0.0
        for feats_fake_real in [
            [mpd_feats_fake, mpd_feats_real],
            [msd_feats_fake, msd_feats_real],
        ]:
            for fake, real in zip(feats_fake_real[0], feats_fake_real[1]):
                for sub_fake, sub_real in zip(fake, real):
                    loss += nn.functional.l1_loss(sub_fake, sub_real)
        return loss
