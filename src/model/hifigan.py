import torch
import torch.nn as nn


class HiFiGAN(nn.Module):
    """
    HiFiGAN implementation based on https://arxiv.org/abs/2010.05646.
    """

    def __init__(
        self, Generator, MultiScaleDiscriminator=None, MultiPeriodDiscriminator=None
    ):
        super().__init__()
        self.Generator = Generator
        if MultiScaleDiscriminator is not None:
            self.MultiScaleDiscriminator = MultiScaleDiscriminator
            self.MultiPeriodDiscriminator = MultiPeriodDiscriminator

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
