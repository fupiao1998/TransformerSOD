import torch
import torch.nn as nn
import torch.nn.functional as F
from model.blocks.base_blocks import BasicConv2d
from model.blocks.rcab_block import RCAB
from model.neck.neck_blocks import ASPP_Module


class rcab_decoder(torch.nn.Module):
    def __init__(self, option):
        super(rcab_decoder, self).__init__()
        self.channel_size = option['neck_channel']
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_reformat_2 = BasicConv2d(in_planes=self.channel_size*2, out_planes=self.channel_size, kernel_size=1)
        self.conv_reformat_3 = BasicConv2d(in_planes=self.channel_size*2, out_planes=self.channel_size, kernel_size=1)
        self.conv_reformat_4 = BasicConv2d(in_planes=self.channel_size*2, out_planes=self.channel_size, kernel_size=1)
        
        self.racb_5, self.racb_4 = RCAB(self.channel_size*2), RCAB(self.channel_size*2)
        self.racb_3, self.racb_2 = RCAB(self.channel_size*2), RCAB(self.channel_size*2)

        self.layer5 = self._make_pred_layer(ASPP_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, self.channel_size*2)
        self.layer6 = self._make_pred_layer(ASPP_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, self.channel_size*2)
        self.layer7 = self._make_pred_layer(ASPP_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, self.channel_size*2)
        self.layer8 = self._make_pred_layer(ASPP_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, self.channel_size*2)
        self.layer9 = self._make_pred_layer(ASPP_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, self.channel_size*1)

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, features):
        x1, x2, x3, x4, x5 = features[0], features[1], features[2], features[3], features[4]

        output1 = F.upsample(self.layer9(x5), scale_factor=32, mode='bilinear', align_corners=True)

        feat_cat = torch.cat((x4, x5), 1)
        feat_cat = self.racb_2(feat_cat)
        output2 = F.upsample(self.layer8(feat_cat), scale_factor=32, mode='bilinear', align_corners=True)
        feat2 = self.conv_reformat_2(feat_cat)

        feat_cat = torch.cat((x3, self.upsample2(feat2)), 1)
        feat_cat = self.racb_3(feat_cat)
        output3 = F.upsample(self.layer7(feat_cat), scale_factor=16, mode='bilinear', align_corners=True)
        feat3 = self.conv_reformat_3(feat_cat)

        feat_cat = torch.cat((x2, self.upsample2(feat3)), 1)
        feat_cat = self.racb_4(feat_cat)
        output4 = F.upsample(self.layer6(feat_cat), scale_factor=8, mode='bilinear', align_corners=True)
        feat4 = self.conv_reformat_4(feat_cat)

        feat_cat = torch.cat((x1, self.upsample2(feat4)), 1)
        feat_cat = self.racb_5(feat_cat)
        output5 = F.upsample(self.layer5(feat_cat), scale_factor=4, mode='bilinear', align_corners=True)

        return [output1, output2, output3, output4, output5]
