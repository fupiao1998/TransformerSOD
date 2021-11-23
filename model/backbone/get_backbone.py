import torch


def get_backbone(option):
    if option['backbone'].lower() == 'swin':
        from model.backbone.swin import SwinTransformer
        backbone = SwinTransformer(img_size=option['trainsize'], embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32], window_size=12)
        pretrained_dict = torch.load(option['pretrain'])["model"]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in backbone.state_dict()}
        backbone.load_state_dict(pretrained_dict)
        channel_list = [128, 256, 512, 1024]
    elif option['backbone'].lower() == 'r50':
        from model.backbone.resnet import ResNet50Backbone
        backbone = ResNet50Backbone()
        channel_list = [256, 512, 1024, 2048]
    elif option['backbone'].lower() == 'dpt':
        from model.backbone.DPT import DPT
        backbone = DPT().cuda()
        channel_list = [256, 512, 768, 768]

    return backbone, channel_list
