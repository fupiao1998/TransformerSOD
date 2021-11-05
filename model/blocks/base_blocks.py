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
        # 2. Activation layer // only suport ReLU now
        if self.act is not None:
            x = self.act(x)
        return x
