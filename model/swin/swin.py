import torch
from .decoder import DepthDecoder, DepthDecoderUp
from .swin_encoder import SwinTransformer


class Swin(torch.nn.Module):
    def __init__(self, img_size, use_attention=False):
        super(Swin, self).__init__()

        self.encoder = SwinTransformer(img_size=img_size, 
                                       embed_dim=128,
                                       depths=[2,2,18,2],
                                       num_heads=[4,8,16,32],
                                       window_size=12
                                       )
        pretrained_dict = torch.load('model/swin/swin_base_patch4_window12_384.pth')["model"]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(pretrained_dict)
        
        self.decoder = DepthDecoder(use_attention=use_attention)

    def forward(self, x):
        features = self.encoder(x)
        # List: [8, 128, 96, 96], [8, 256, 48, 48], [8, 512, 24, 24], [8, 1024, 12, 12], [8, 1024, 12, 12]
        outputs = self.decoder(features)

        return outputs
