import torch
import torch.nn as nn
import math
from utils.path_hyperparameter import ph


class Conv_BN_ReLU(nn.Module):
    """ Basic convolution."""

    def __init__(self, in_channel, out_channel, kernel, stride):
        super().__init__()
        self.conv_bn_relu = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=kernel,
                                                    padding=kernel // 2, bias=False, stride=stride),
                                          nn.BatchNorm2d(out_channel),
                                          nn.ReLU(inplace=True),
                                          )

    def forward(self, x):
        output = self.conv_bn_relu(x)

        return output


class Encoder_Block(nn.Module):
    """ Basic block in encoder"""

    def __init__(self, in_channel, out_channel):
        super().__init__()

        assert out_channel == in_channel * 2, 'the out_channel is not in_channel*2 in encoder block'
        self.conv = nn.Sequential(
            Conv_BN_ReLU(in_channel=in_channel, out_channel=out_channel, kernel=3, stride=2),
            Conv_BN_ReLU(in_channel=out_channel, out_channel=out_channel, kernel=3, stride=1),
            Conv_BN_ReLU(in_channel=out_channel, out_channel=out_channel, kernel=3, stride=1),
        )

    def forward(self, x):
        output = self.conv(x)

        return output


class Decoder_Block(nn.Module):
    """Basic block in decoder."""

    def __init__(self, in_channel, out_channel):
        super().__init__()

        assert out_channel == in_channel // 2, 'the out_channel is not in_channel//2 in decoder block'
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                            kernel_size=1, padding=0, bias=False),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU(inplace=True),
                                  )

    def forward(self, de):
        de = self.up(de)
        output = self.conv(de)

        return output
