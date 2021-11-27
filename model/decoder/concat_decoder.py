import torch
import torch.nn as nn
from model.blocks.base_blocks import FeatureFusionBlock, SimpleHead


class simple_decoder(torch.nn.Module):
    def __init__(self, option):
        super(simple_decoder, self).__init__()
        self.channel_size = option['neck_channel']
        self.deep_sup = option['deep_sup']
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.short_connect_with_decode_module = nn.ModuleList()
        for _ in range(4):
            self.short_connect_with_decode_module.append(FeatureFusionBlock(self.channel_size))

        self.head_up_2 = SimpleHead(channel=self.channel_size, rate=2)
        if self.deep_sup:
            self.head_up_4 = SimpleHead(channel=self.channel_size, rate=4)
            self.head_up_8 = SimpleHead(channel=self.channel_size, rate=8)
            self.head_up_16 = SimpleHead(channel=self.channel_size, rate=16)

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, features):
        features = features[::-1]

        conv_feat_list = []
        for i in range(len(features)):
            if i == 0:
                conv_feat = self.short_connect_with_decode_module[i](features[i])
            else:
                conv_feat = self.short_connect_with_decode_module[i](conv_feat, features[i])
            conv_feat_list.append(conv_feat)

        output4 = self.head_up_2(conv_feat_list[3])        

        if self.deep_sup:
            output1 = self.head_up_16(conv_feat_list[0])
            output2 = self.head_up_8(conv_feat_list[1])
            output3 = self.head_up_4(conv_feat_list[2])
            return [output1, output2, output3, output4]
        else:
            return [output4]