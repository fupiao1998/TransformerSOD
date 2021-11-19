import torch.nn as nn


def get_depth_module(option):
    depth_module = nn.ModuleDict()
    if option['task'].lower() != 'rgbd-sod':
        depth_module = None
    else:
        if option['fusion'].lower() == 'early':
            from model.depth_module.early_fusion import early_fusion_conv
            depth_module['head'] = early_fusion_conv()
        elif option['fusion'].lower() == 'late':
            from model.depth_module.depth_feature import depth_feature
            from model.depth_module.feature_fusion import feature_fusion
            depth_module['feature'] = depth_feature(in_planes=128, out_planes=option['neck_channel'])
            depth_module['fusion'] = feature_fusion(option=option)

    return depth_module
