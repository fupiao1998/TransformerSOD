import torch
import torch.nn as nn
from model.blocks.base_blocks import BasicConv2d


class early_fusion_conv(nn.Module):
    def __init__(self, in_channel=4, out_channel=3):
        super(early_fusion_conv, self).__init__()
        self.reduce = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x, depth):
        return self.reduce(torch.cat([x, depth], dim=1))
