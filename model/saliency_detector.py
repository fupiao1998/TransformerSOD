import torch
import torch.nn as nn
from model.backbone.get_backbone import get_backbone
from model.neck.get_neck import get_neck
from model.decoder.get_decoder import get_decoder
from model.depth_module.get_depth_module import get_depth_module


class sod_model(torch.nn.Module):
    def __init__(self, option):
        super(sod_model, self).__init__()

        self.backbone, self.channel_list = get_backbone(option)
        self.neck = get_neck(option, self.channel_list)
        self.decoder = get_decoder(option)
        self.depth_module = get_depth_module(option)

    def forward(self, x, depth=None):
        if depth is not None:
            if 'head' in self.depth_module.keys():
                x = self.depth_module['head'](x, depth)
            elif 'feature' in self.depth_module.keys():
                depth_features = self.depth_module['feature'](depth)

        backbone_features = self.backbone(x)
        neck_features = self.neck(backbone_features)
        if depth is not None and 'fusion' in self.depth_module.keys():
            neck_features = self.depth_module['fusion'](neck_features, depth_features)

        outputs = self.decoder(neck_features)

        return outputs


class Discriminator(nn.Module):
    def __init__(self, ndf):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(4, ndf, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Conv2d(ndf, 1, kernel_size=3, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.bn1 = nn.BatchNorm2d(ndf)
        self.bn2 = nn.BatchNorm2d(ndf)
        self.bn3 = nn.BatchNorm2d(ndf)
        self.bn4 = nn.BatchNorm2d(ndf)

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.leaky_relu(self.bn4(self.conv4(x)))

        x = self.classifier(x)
        return x
