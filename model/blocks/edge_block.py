import torch
import torch.nn as nn
import torch.nn.functional as F
from model.blocks.rcab_block import RCAB


class edge_module(nn.Module):
    def __init__(self, in_channels=[256, 256, 256], mid_feat=32):
        super(edge_module, self).__init__()
        self.in_channels = in_channels
        self.conv_list = nn.ModuleList()
        for in_channel in self.in_channels:
            self.conv_list.append(self.make_stage_layers(in_channel, mid_feat))

        self.out_conv = nn.Conv2d(mid_feat*3, 1, kernel_size=3, padding=1)
        self.rcab = RCAB(mid_feat*3)

    def make_stage_layers(self, in_channel, mid_feat):
        block = []
        block.append(nn.Conv2d(in_channel, mid_feat, 1))
        block.append(nn.ReLU(inplace=True))
        block.append(nn.Conv2d(mid_feat, mid_feat, 3, padding=1))
        block.append(nn.ReLU(inplace=True))

        return nn.Sequential(*block)

    def forward(self, in_feature):
        assert len(in_feature) == len(self.in_channels)
        _, _, h, w = in_feature[0].shape
        edge_out = []
        for i, edge_conv_block in enumerate(self.conv_list):
            edge_conv_out = edge_conv_block(in_feature[i])
            edge_out.append(F.interpolate(edge_conv_out, size=(h, w), mode='bilinear', align_corners=True))

        edge = torch.cat(edge_out, dim=1)
        edge = self.rcab(edge)
        edge = self.out_conv(edge)
        return edge
