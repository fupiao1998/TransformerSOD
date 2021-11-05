import torch.nn as nn
from model.blocks.base_blocks import BasicConv2d


class ASPP_Module(nn.Module):
    def __init__(self, dilation_series, padding_series, out_channel, input_channel):
        super(ASPP_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(BasicConv2d(input_channel, out_channel, kernel_size=3, stride=1, padding=padding, dilation=dilation, norm=False))

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out


class DimReduce(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DimReduce, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)
