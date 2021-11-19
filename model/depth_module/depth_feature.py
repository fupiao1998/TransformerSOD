import torch
import torch.nn as nn
from model.blocks.base_blocks import BasicConv2d, ResidualBlock



class depth_feature(torch.nn.Module):
    def __init__(self, in_planes, out_planes):
        super(depth_feature, self).__init__()

        self.in_planes = in_planes
        self.depth_ConvReLU = BasicConv2d(in_planes=1, out_planes=64, kernel_size=3, stride=1, padding=1, norm=False, act=True)
        self.depth_ConvReLU1x1 = BasicConv2d(in_planes=64, out_planes=self.in_planes, kernel_size=1, stride=1, padding=0, norm=False, act=True)
        self.depth_layer0 = self._make_layer(self.in_planes, stride=2)
        self.depth_layers = nn.ModuleList()
        for stride in [2, 2, 2, 2, 1]:
            self.depth_layers.append(self._make_layer(out_planes, stride=stride))

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, stride=stride)
        layer2 = ResidualBlock(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim        
        return nn.Sequential(*layers)

    def forward(self, depth):
        depth_feat_0 = self.depth_ConvReLU(depth)
        depth_feat_0 = self.depth_ConvReLU1x1(depth_feat_0)
        depth_feat_0 = self.depth_layer0(depth_feat_0)

        out_list = []
        for depth_layer in self.depth_layers:
            out_list.append(depth_layer(depth_feat_0))
            depth_feat_0 = out_list[-1]

        return out_list


if __name__ == "__main__":
    model = depth_feature(in_planes=128, out_planes=128)
    print("[INFO]: model have {:.4f}Mb paramerters in total".format(sum(x.numel()/1e6 for x in model.parameters())))
    x = torch.rand(4, 1, 384, 384)
    y = model(x)
