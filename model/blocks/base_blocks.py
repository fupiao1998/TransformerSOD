import torch.nn as nn


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, norm=True, act=False):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        if norm: 
            self.norm = nn.BatchNorm2d(out_planes)
        else:
            self.norm = None
        
        if act: 
            self.act = nn.ReLU()   
        else:
            self.act = None     
        
    def forward(self, x):
        # 1. Conv layer
        x = self.conv(x)
        # 2. Norm layer // only suport BN now
        if self.norm is not None:
            x = self.norm(x)
        # 3. Activation layer // only suport ReLU now
        if self.act is not None:
            x = self.act(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(ResidualBlock, self).__init__()
  
        self.ConvBNReLU1 = BasicConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, norm=True, act=True)
        self.ConvBNReLU2 = BasicConv2d(planes, planes, kernel_size=3, padding=1, norm=True, act=True)

        if stride == 1:
            self.downsample = None
        else:    
            self.downsample = BasicConv2d(in_planes, planes, kernel_size=1, stride=stride, norm=True)

    def forward(self, x):
        y = x
        y = self.ConvBNReLU1(y)
        y = self.ConvBNReLU2(y)

        if self.downsample is not None:
            x = self.downsample(x)

        return x + y


class FeatureFusionBlock(nn.Module):

    def __init__(self, features):
        super(FeatureFusionBlock, self).__init__()
        self.resConfUnit1 = ResidualBlock(features, features)
        self.resConfUnit2 = ResidualBlock(features, features)

    def forward(self, *xs):
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])
            output = self.resConfUnit2(output)
            output = nn.functional.interpolate(output, scale_factor=2, mode="bilinear", align_corners=True)

            return output
        else:
            output = self.resConfUnit2(output)
            output = nn.functional.interpolate(output, scale_factor=2, mode="bilinear", align_corners=True)

            return output


class SimpleHead(nn.Module):
    def __init__(self, channel, rate):
        super(SimpleHead, self).__init__()
        self.upsample = nn.Upsample(scale_factor=rate, mode='bilinear', align_corners=True)
  
        self.output_conv = nn.Sequential(
            nn.Conv2d(channel, channel//2, kernel_size=3, stride=1, padding=1),
            self.upsample,
            nn.Conv2d(channel//2, channel//4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(channel//4, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.output_conv(x)
