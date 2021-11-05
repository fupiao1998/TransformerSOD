import torch
import torch.nn as nn
from model.neck.neck_blocks import ASPP_Module, DimReduce


class aspp_neck(torch.nn.Module):
    def __init__(self, in_channel_list=[128, 256, 512, 1024, 1024], out_channel=256):
        super(aspp_neck, self).__init__()

        self.out_channel = out_channel
        self.in_channel_list = in_channel_list
        self.conv_list = nn.ModuleList()
        for in_channel_i in self.in_channel_list:
            self.conv_list.append(ASPP_Module([3, 6, 12, 18], [3, 6, 12, 18], self.out_channel, in_channel_i))
        # swin<->in_channel_list: [128, 256, 512, 1024, 1024]

    def forward(self, features):
        out_list = []
        for i, conv in enumerate(self.conv_list):
            out_list.append(conv(features[i]))

        return out_list


class basic_neck(torch.nn.Module):
    def __init__(self, in_channel_list=[128, 256, 512, 1024, 1024], out_channel=256):
        super(basic_neck, self).__init__()

        self.out_channel = out_channel
        self.in_channel_list = in_channel_list
        self.conv_list = nn.ModuleList()
        for in_channel_i in self.in_channel_list:
            self.conv_list.append(DimReduce(in_channel_i, self.out_channel).cuda())
        # swin<->in_channel_list: [128, 256, 512, 1024, 1024]

    def forward(self, features):
        out_list = []
        for i, conv in enumerate(self.conv_list):
            out_list.append(conv(features[i]))

        return out_list
