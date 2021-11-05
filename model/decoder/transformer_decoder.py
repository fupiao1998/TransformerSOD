import torch
import torch.nn as nn
import torch.nn.functional as F

from model.decoder.trans_blocks.basic import _ConvBNReLU, SeparableConv2d
from model.decoder.trans_blocks.transformer import VisionTransformer




# vit_params = cfg.MODEL.TRANS2Seg
# vit_params['decoder_feat_HxW'] = c4_HxW
# self.transformer_head = TransformerHead(vit_params, c1_channels=c1_channels, c4_channels=c4_channels, hid_dim=hid_dim)

'''
cfg.MODEL.TRANS2Seg.embed_dim = 256
cfg.MODEL.TRANS2Seg.depth = 4
cfg.MODEL.TRANS2Seg.num_heads = 8
cfg.MODEL.TRANS2Seg.mlp_ratio = 3.
cfg.MODEL.TRANS2Seg.hid_dim = 64
'''
class Transformer(nn.Module):
    def __init__(self, vit_params, c4_channels=2048):
        super().__init__()
        last_channels = vit_params['embed_dim']
        self.vit = VisionTransformer(input_dim=c4_channels,
                                     embed_dim=last_channels,
                                     depth=vit_params['depth'],
                                     num_heads=vit_params['num_heads'],
                                     mlp_ratio=vit_params['mlp_ratio'],
                                     decoder_feat_HxW=vit_params['decoder_feat_HxW'])

    def forward(self, x):
        n, _, h, w = x.shape
        x = self.vit.hybrid_embed(x)

        cls_token, x = self.vit.forward_encoder(x)

        attns_list = self.vit.forward_decoder(x)

        x = x.reshape(n, h, w, -1).permute(0, 3, 1, 2)
        return x, attns_list


class transformer_decoder(nn.Module):
    def __init__(self, vit_params, c1_channels=256, c4_channels=2048, hid_dim=64, norm_layer=nn.BatchNorm2d):
        super().__init__()

        last_channels = vit_params['embed_dim']
        nhead = vit_params['num_heads']

        self.transformer = Transformer(vit_params, c4_channels=c4_channels)

        self.conv_c1 = _ConvBNReLU(c1_channels, hid_dim, 1, norm_layer=norm_layer)

        self.lay1 = SeparableConv2d(last_channels+nhead, hid_dim, 3, norm_layer=norm_layer, relu_first=False)
        self.lay2 = SeparableConv2d(hid_dim, hid_dim, 3, norm_layer=norm_layer, relu_first=False)
        self.lay3 = SeparableConv2d(hid_dim, hid_dim, 3, norm_layer=norm_layer, relu_first=False)

        self.pred = nn.Conv2d(hid_dim, 1, 1)


    def forward(self, features):
        c1, c2, c3, c4, c5 = features[0], features[1], features[2], features[3], features[4]
        feat_enc, attns_list = self.transformer(c4)
        
        attn_map = attns_list[-1]
        B, nclass, nhead, _ = attn_map.shape
        _, _, H, W = feat_enc.shape
        attn_map = attn_map.reshape(B*nclass, nhead, H, W)
        x = torch.cat([_expand(feat_enc, nclass), attn_map], 1)

        x = self.lay1(x)
        x = self.lay2(x)

        size = c1.size()[2:]
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        c1 = self.conv_c1(c1)
        x = x + _expand(c1, nclass)

        x = self.lay3(x)
        x = self.pred(x).reshape(B, nclass, size[0], size[1])
        x = F.upsample(x, scale_factor=4, mode='bilinear', align_corners=True)

        return [x]


def _expand(x, nclass):
    return x.unsqueeze(1).repeat(1, nclass, 1, 1, 1).flatten(0, 1)


if __name__ == '__main__':
    '''
    # vit small
    vit_params = {}
    vit_params['embed_dim'] = 256
    vit_params['depth'] = 4
    vit_params['num_heads'] = 8
    vit_params['mlp_ratio'] = 3.0
    vit_params['hid_dim'] = 64
    vit_params['decoder_feat_HxW'] = 12*12
    '''
    vit_params = {}
    vit_params['embed_dim'] = 128
    vit_params['depth'] = 1
    vit_params['num_heads'] = 4
    vit_params['mlp_ratio'] = 2.0
    vit_params['hid_dim'] = 32
    vit_params['decoder_feat_HxW'] = 12*12
    c4 = torch.randn(1, 128, 12, 12).cuda()
    c1 = torch.randn(1, 128, 96, 96).cuda()
    model = transformer_decoder(vit_params, c1_channels=128, c4_channels=128).cuda()
    print("[INFO]: TransformerHead have {:.4f}Mb paramerters in total".format(sum(x.numel()/1e6 for x in model.parameters())))
    out = model(c4, c1)
    print(out.shape)
