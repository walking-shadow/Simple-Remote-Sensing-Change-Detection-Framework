import torch.nn as nn
from models.dpcd_parts import (Conv_BN_ReLU, Encoder_Block, Decoder_Block)


class DPCD(nn.Module):
    def __init__(self):
        super().__init__()

        channel_list = [32, 64, 128, 256, 512]
        # encoder
        self.en_block1 = nn.Sequential(Conv_BN_ReLU(in_channel=3, out_channel=channel_list[0], kernel=3, stride=1))
        self.en_block2 = Encoder_Block(in_channel=channel_list[0], out_channel=channel_list[1])
        self.en_block3 = Encoder_Block(in_channel=channel_list[1], out_channel=channel_list[2])
        self.en_block4 = Encoder_Block(in_channel=channel_list[2], out_channel=channel_list[3])
        self.en_block5 = Encoder_Block(in_channel=channel_list[3], out_channel=channel_list[4])

        # decoder
        self.de_block1 = Decoder_Block(in_channel=channel_list[4], out_channel=channel_list[3])
        self.de_block2 = Decoder_Block(in_channel=channel_list[3], out_channel=channel_list[2])
        self.de_block3 = Decoder_Block(in_channel=channel_list[2], out_channel=channel_list[1])
        self.de_block4 = Decoder_Block(in_channel=channel_list[1], out_channel=channel_list[0])

        self.conv_out_change = nn.Conv2d(channel_list[0], 1, kernel_size=7, stride=1, padding=3)

    def forward(self, t1, t2):
        # encoder
        t1_1 = self.en_block1(t1)
        t2_1 = self.en_block1(t2)

        t1_2 = self.en_block2(t1_1)
        t2_2 = self.en_block2(t2_1)

        t1_3 = self.en_block3(t1_2)
        t2_3 = self.en_block3(t2_2)

        t1_4 = self.en_block4(t1_3)
        t2_4 = self.en_block4(t2_3)

        t1_5 = self.en_block5(t1_4)
        t2_5 = self.en_block5(t2_4)

        de = t1_5 + t2_5
        de = self.de_block1(de)
        de = self.de_block2(de)
        de = self.de_block3(de)
        de = self.de_block4(de)
        change_out = self.conv_out_change(de)

        return change_out
