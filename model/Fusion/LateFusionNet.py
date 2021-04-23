import torch
import torch.nn as nn
import torch.nn.functional as F
from model.DPT.blocks import (
    FeatureFusionBlock,
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_vit,
)
from model.DPT.DPT import BaseModel, _make_fusion_block
from model.ResNet.ResNet import B2_ResNet


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    # paper: Image Super-Resolution Using Very DeepResidual Channel Attention Networks
    # input: B*C*H*W
    # output: B*C*H*W
    def __init__(
        self, n_feat, kernel_size=3, reduction=16,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: 
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: 
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand == True:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0],
        out_shape1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3],
        out_shape4,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )

    return scratch


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class DPT(BaseModel):
    def __init__(
        self,
        head_1, head_2,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
    ):

        super(DPT, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            True,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )

        self.scratch_resnet = _make_scratch([256, 512, 768, 768], features, groups=1, expand=False)
        self.scratch_resnet.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch_resnet.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch_resnet.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch_resnet.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = head_1
        self.scratch_resnet.output_conv = head_2
        self.fuse_output_conv = conv1x1(256, 1, stride=1)

        self.resnet = B2_ResNet()
        self.resnet_backbone_conv1 = conv1x1(1024, 768, stride=1)
        self.resnet_backbone_conv2 = conv1x1(2048, 768, stride=1)
        self.fusion_conv = conv3x3(2, 256, stride=1)
        self.racb = RCAB(256)

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward_resnet(self, model, x):
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x1 = model.layer1(x)  # 256 x 64 x 64
        x2 = model.layer2(x1)  # 512 x 32 x 32

        x3 = model.layer3_1(x2)  # 1024 x 16 x 16
        x4 = model.layer4_1(x3)  # 2048 x 8 x 8

        x3 = self.resnet_backbone_conv1(x3)
        x4 = self.resnet_backbone_conv2(x4)

        return x1, x2, x3, x4

    def forward(self, x):
        # x = torch.Size([1, 3, 384, 672])
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)
        ### Encoder of VIT and ResNet
        layer_1_vit, layer_2_vit, layer_3_vit, layer_4_vit = forward_vit(self.pretrained, x)
        # [4, 256, 88, 88], [4, 512, 44, 44], [4, 768, 22, 22], [4, 768, 11, 11]
        layer_1_resnet, layer_2_resnet, layer_3_resnet, layer_4_resnet = self.forward_resnet(self.resnet, x)
        # [4, 256, 88, 88], [4, 512, 44, 44], [4, 1024, 22, 22], [4, 2048, 11, 11]

        ### Short connections between encoder and decoder
        layer_1_rn_vit = self.scratch.layer1_rn(layer_1_vit)
        layer_2_rn_vit = self.scratch.layer2_rn(layer_2_vit)
        layer_3_rn_vit = self.scratch.layer3_rn(layer_3_vit)
        layer_4_rn_vit = self.scratch.layer4_rn(layer_4_vit)

        layer_1_rn_resnet = self.scratch_resnet.layer1_rn(layer_1_resnet)
        layer_2_rn_resnet = self.scratch_resnet.layer2_rn(layer_2_resnet)
        layer_3_rn_resnet = self.scratch_resnet.layer3_rn(layer_3_resnet)
        layer_4_rn_resnet = self.scratch_resnet.layer4_rn(layer_4_resnet)

        ### Decoders
        path_4_vit = self.scratch.refinenet4(layer_4_rn_vit)
        path_3_vit = self.scratch.refinenet3(path_4_vit, layer_3_rn_vit)
        path_2_vit = self.scratch.refinenet2(path_3_vit, layer_2_rn_vit)
        path_1_vit = self.scratch.refinenet1(path_2_vit, layer_1_rn_vit)

        path_4_resnet = self.scratch_resnet.refinenet4(layer_4_rn_resnet)
        path_3_resnet = self.scratch_resnet.refinenet3(path_4_resnet, layer_3_rn_resnet)
        path_2_resnet = self.scratch_resnet.refinenet2(path_3_resnet, layer_2_rn_resnet)
        path_1_resnet = self.scratch_resnet.refinenet1(path_2_resnet, layer_1_rn_resnet)
        
        out_vit = self.scratch.output_conv(path_1_vit)
        out_resnet = self.scratch_resnet.output_conv(path_1_resnet)

        concat_result = torch.cat([out_vit, out_resnet], 1)
        fusion_feature = self.racb(self.fusion_conv(concat_result))
        out_fusion = self.fuse_output_conv(fusion_feature)

        return [out_vit, out_resnet, out_fusion]


class LateFusionSegmentationModel(DPT):
    def __init__(self, num_classes, path=None, **kwargs):

        features = kwargs["features"] if "features" in kwargs else 256

        kwargs["use_bn"] = True

        head_1 = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, num_classes, kernel_size=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )
        head_2 = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, num_classes, kernel_size=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        super().__init__(head_1, head_2, **kwargs)

        self.auxlayer = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, num_classes, kernel_size=1),
        )

        if path is not None:
            self.load(path)
