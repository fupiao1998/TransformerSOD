import torch
import torch.nn as nn
from model.blocks.rcab_block import RCAB
from model.blocks.base_blocks import SimpleHead
from model.neck.neck_blocks import ASPP_Module


class concat_decoder_deep_sup(torch.nn.Module):
    def __init__(self, option):
        super(concat_decoder_deep_sup, self).__init__()
        self.channel_size = option['neck_channel']
        self.deep_sup = option['deep_sup']

        self.rcab_conv = RCAB(4*self.channel_size)
        self.aspp_head = ASPP_Module(dilation_series=[3, 6, 12, 18], padding_series=[3, 6, 12, 18], 
                                     out_channel=1, input_channel=4*self.channel_size)
        
        self.aspp_head_1 = ASPP_Module(dilation_series=[3, 6, 12, 18], padding_series=[3, 6, 12, 18], 
                                       out_channel=1, input_channel=self.channel_size)
        self.aspp_head_2 = ASPP_Module(dilation_series=[3, 6, 12, 18], padding_series=[3, 6, 12, 18], 
                                       out_channel=1, input_channel=self.channel_size)
        self.aspp_head_3 = ASPP_Module(dilation_series=[3, 6, 12, 18], padding_series=[3, 6, 12, 18], 
                                       out_channel=1, input_channel=self.channel_size)
        self.aspp_head_4 = ASPP_Module(dilation_series=[3, 6, 12, 18], padding_series=[3, 6, 12, 18], 
                                       out_channel=1, input_channel=self.channel_size)

    def forward(self, features):
        up_feat_list = []
        for i, feat in enumerate(features):
            up_feat_list.append(nn.functional.interpolate(feat, scale_factor=(2**i), mode='bilinear', align_corners=True))
            
        pred_1 = nn.functional.interpolate(self.aspp_head_1(features[3]), scale_factor=32, mode='bilinear', align_corners=True)
        pred_2 = nn.functional.interpolate(self.aspp_head_2(features[2]), scale_factor=16, mode='bilinear', align_corners=True)
        pred_3 = nn.functional.interpolate(self.aspp_head_3(features[1]), scale_factor=8, mode='bilinear', align_corners=True)
        pred_4 = nn.functional.interpolate(self.aspp_head_4(features[0]), scale_factor=4, mode='bilinear', align_corners=True)

        feat = torch.cat(up_feat_list, dim=1)
        pred = nn.functional.interpolate(self.aspp_head(self.rcab_conv(feat)), scale_factor=4, mode='bilinear', align_corners=True)

        return [pred_1, pred_2, pred_3, pred_4, pred]
