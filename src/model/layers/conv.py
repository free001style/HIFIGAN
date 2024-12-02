import torch
import torch.nn as nn


class Conv(nn.Module):
    """
    Convolutional layer with normalization layer, activation function and weights initialization.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=None,
        dilation=1,
        groups=1,
        bias=False,
        normalization=None,
        activation=nn.Identity(),
        is_2d=True,
    ):
        """
        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (int | tuple(int)): size of convolutional kernel.
            stride (int | tuple(int)): stride of convolution.
            padding (str | int | tuple(int)): padding of convolutional layer.
            dilation (int | tuple(int)): dilation of convolutional layer.
            groups (int): if equal to out_channels, depth-wise layer is used.
            bias (bool): bias of convolutional layer.
            normalization (nn.Module, optional): normalization layer. if None, no normalization will be applied.
            activation (nn.Module, optional): activation function. if None, no activation will be applied.
            is_2d (bool): if True, 2D convolution is used, otherwise 1D convolution.
        """
        super().__init__()
        self.conv = nn.Conv2d if is_2d else nn.Conv1d
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = self.conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        # nn.init.xavier_uniform_(self.conv.weight, nn.init.calculate_gain("leaky_relu"))
        self.activation = activation
        if normalization is not None:
            self.conv = normalization(self.conv)

    def forward(self, x):
        return self.activation(self.conv(x))
