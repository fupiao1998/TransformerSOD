import torch
import torch.nn as nn


class DepthAttentionBlock(nn.Module):
    def __init__(self, channel, hidden_channel):
        super(DepthAttentionBlock, self).__init__()

        self.inplanes = channel
        self.planes = hidden_channel
        self.conv_mask = nn.Conv2d(self.inplanes, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.softmax_channel = nn.Softmax(dim=1)
        self.channel_mul_conv = nn.Sequential(
            nn.Conv2d(self.inplanes, self.planes, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.LayerNorm([self.planes, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.planes, self.inplanes, kernel_size=(1, 1), stride=(1, 1), bias=False),
        )

    def spatial_pool(self, depth_feature):
        batch, channel, height, width = depth_feature.size()
        input_x = depth_feature
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(depth_feature)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        # context attention
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x, depth_feature):
        # [N, C, 1, 1]
        context = self.spatial_pool(depth_feature)
        # [N, C, 1, 1]
        channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
        # channel-wise attention
        out1 = torch.sigmoid(depth_feature * channel_mul_term)
        # fusion
        out = x * out1

        return torch.sigmoid(out)


class DepthRefineBlock(nn.Module):
    def __init__(self, channel=256):
        super(DepthRefineBlock, self).__init__()
        self.conv_refine1 = nn.Conv2d(channel, channel, 3, padding=1)
        self.bn_refine1 = nn.BatchNorm2d(channel, eps=1e-05, momentum=0.1, affine=True)

        self.conv_refine2 = nn.Conv2d(channel, channel, 3, padding=1)
        self.bn_refine2 = nn.BatchNorm2d(channel, eps=1e-05, momentum=0.1, affine=True)
        self.prelu = nn.PReLU()

        self.conv_fuse = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_out = nn.Conv2d(channel, channel, 3, padding=1)

    def forward(self, img_feat, depth_feat):
        assert img_feat.shape[1] == depth_feat.shape[1], 'Input channel of image feature and depth feature must the same!'
        depth_feat_1 = self.prelu(self.bn_refine1(self.conv_refine1(depth_feat)))
        depth_feat_2 = self.prelu(self.bn_refine2(self.conv_refine2(depth_feat_1)))

        fused_feat = img_feat + depth_feat_2
        fused_feat_skip = self.prelu(self.conv_fuse(fused_feat)) + fused_feat
        output = self.conv_out(fused_feat_skip)

        return output


class feature_fusion(nn.Module):
    def __init__(self, option):
        super(feature_fusion, self).__init__()
        self.channel = option['neck_channel']
        self.fusion_blocks = nn.ModuleList()
        for i in range(5):
            if option['fusion_method'] == 'refine':
                self.fusion_blocks.append(DepthRefineBlock(channel=self.channel))
            elif option['fusion_method'] == 'attention':
                self.fusion_blocks.append(DepthAttentionBlock(channel=self.channel, hidden_channel=self.channel//4*3))

    def forward(self, img_feat_list, depth_feat_list):
        assert len(img_feat_list) == len(depth_feat_list), 'Input channel of image feature and depth feature must the same!'
        fusion_feat_list = []
        for block, img_feat, depth_feat in zip(self.fusion_blocks, img_feat_list, depth_feat_list):
            fusion_feat_list.append(block(img_feat, depth_feat))

        return fusion_feat_list
