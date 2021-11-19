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
        self.relu = nn.ReLU(inplace=True)

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

        return self.relu(x+y)
