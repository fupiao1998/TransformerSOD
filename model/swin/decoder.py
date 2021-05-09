# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from .layers import *
from model.blocks.attention import AttentionConv


class DepthDecoder(nn.Module):
    # def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
    def __init__(self, scales=range(4), num_output_channels=1, use_skips=True, use_multi_scale=True, use_attention=False):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.use_multi_scale = use_multi_scale
        self.use_attention = use_attention

        # self.num_ch_enc = np.array([96, 192, 384, 768])
        self.num_ch_enc = np.array([128, 256, 512, 1024, 1024])
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        # self.num_ch_dec = np.array([96, 192, 384, 768])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
        # for i in range(3, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            # num_ch_in = self.num_ch_enc[-1] if i == 3 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        if self.use_attention:
            self.att_conv1 = AttentionConv(in_channels=1280, out_channels=1280, kernel_size=3, padding=1, groups=4)
            self.att_conv2 = AttentionConv(in_channels=640, out_channels=640, kernel_size=3, padding=1, groups=4)
        else:
            self.att_conv1, self.att_conv2 = None, None

    def forward(self, input_features):
        self.outputs = {}
        output_list = []

        # decoder
        x = upsample(input_features[-1])
        for i in range(4, -1, -1):
        # for i in range(3, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            if i == 4:
                x = [x]
            else:
                x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [upsample(input_features[i - 1])]
            x = torch.cat(x, 1)
            if self.att_conv1 is not None and i == 4:
                x = self.att_conv1(x)
            elif self.att_conv2 is not None and i == 3:
                x = self.att_conv2(x)

            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                output_list.append(self.convs[("dispconv", i)](x))
        if self.use_multi_scale:
            output_list[0] = F.interpolate(output_list[0], scale_factor=8, mode="bilinear", align_corners=False)
            output_list[1] = F.interpolate(output_list[1], scale_factor=4, mode="bilinear", align_corners=False)
            output_list[2] = F.interpolate(output_list[2], scale_factor=2, mode="bilinear", align_corners=False)
            return output_list
        else:
            return output_list[-1]


class DepthDecoderUp(nn.Module):
    def __init__(self, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoderUp, self).__init__()

        print('upsampling output')

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        # self.num_ch_enc = num_ch_enc
        self.num_ch_enc = np.array([128, 256, 512, 1024, 1024])
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            if i == 4:
                x = [x]
            else:
                x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = upsample(self.sigmoid(self.convs[("dispconv", i)](x)))

        return self.outputs