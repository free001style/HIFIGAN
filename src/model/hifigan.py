import torch
import torch.nn as nn


class HiFiGAN(nn.Module):
    def __init__(
        self, Generator, MultiScaleDiscriminator=None, MultiPeriodDiscriminator=None
    ):
        super().__init__()
        self.Generator = Generator
        if MultiScaleDiscriminator is not None:
            self.MultiScaleDiscriminator = MultiScaleDiscriminator
            self.MultiPeriodDiscriminator = MultiPeriodDiscriminator
