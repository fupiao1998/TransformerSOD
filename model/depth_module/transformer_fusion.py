import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super(SelfAttention, self).__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super(Block, self).__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True), # changed from GELU
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, n_embd, vert_anchors=11, horz_anchors=11, n_head=8, block_exp=1, n_layer=1, seq_len=1, embd_pdrop=0.5, attn_pdrop=0.1, resid_pdrop=0.1):
        super(GPT, self).__init__()
        self.n_embd = n_embd
        self.seq_len = seq_len
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        self.n_views = 1

        # positional embedding parameter (learnable), image + lidar
        self.pos_emb = nn.Parameter(
            torch.zeros(1, (self.n_views + 1) * seq_len * vert_anchors * horz_anchors, n_embd))

        # velocity embedding
        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_exp, attn_pdrop, resid_pdrop) for layer in range(n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)

        self.block_size = seq_len
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # def configure_optimizers(self):
    #     # separate out all parameters to those that will and won't experience regularizing weight decay
    #     decay = set()
    #     no_decay = set()
    #     whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
    #     blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm2d)
    #     for mn, m in self.named_modules():
    #         for pn, p in m.named_parameters():
    #             fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

    #             if pn.endswith('bias'):
    #                 # all biases will not be decayed
    #                 no_decay.add(fpn)
    #             elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
    #                 # weights of whitelist modules will be weight decayed
    #                 decay.add(fpn)
    #             elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
    #                 # weights of blacklist modules will NOT be weight decayed
    #                 no_decay.add(fpn)

    #     # special case the position embedding parameter in the root GPT module as not decayed
    #     no_decay.add('pos_emb')

    #     # create the pytorch optimizer object
    #     param_dict = {pn: p for pn, p in self.named_parameters()}
    #     optim_groups = [
    #         {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
    #         {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    #     ]

    #     return optim_groups

    def forward(self, image_tensor, lidar_tensor):
        raw_image_tensor = image_tensor
        raw_lidar_tensor = lidar_tensor
        image_tensor = F.upsample(image_tensor, size=(self.vert_anchors, self.horz_anchors), mode='bilinear', align_corners=True)
        lidar_tensor = F.upsample(lidar_tensor, size=(self.vert_anchors, self.horz_anchors), mode='bilinear', align_corners=True)
        """
        Args:
            image_tensor (tensor): B*4*seq_len, C, H, W
            lidar_tensor (tensor): B*seq_len, C, H, W
            velocity (tensor): ego-velocity
        """

        bz = lidar_tensor.shape[0] // self.seq_len
        h, w = lidar_tensor.shape[2:4]

        # forward the image model for token embeddings
        image_tensor = image_tensor.view(bz, self.seq_len, -1, h, w)
        lidar_tensor = lidar_tensor.view(bz, self.seq_len, -1, h, w)

        # print(image_tensor.size())
        # print(lidar_tensor.size())

        # pad token embeddings along number of tokens dimension
        token_embeddings = torch.cat([image_tensor, lidar_tensor], dim=1).permute(0, 1, 3, 4, 2).contiguous()
        token_embeddings = token_embeddings.view(bz, -1, self.n_embd)  # (B, an * T, C)

        # project velocity to n_embed
        # [dxli] we don't need veloctiy embeddings anyway.
        # velocity_embeddings = self.vel_emb(velocity.unsqueeze(1)) # (B, C)

        # add (learnable) positional embedding and velocity embedding for all tokens
        x = self.drop(self.pos_emb + token_embeddings)  # + velocity_embeddings.unsqueeze(1)) # (B, an * T, C)
        # x = self.drop(token_embeddings + velocity_embeddings.unsqueeze(1)) # (B, an * T, C)
        x = self.blocks(x)  # (B, an * T, C)
        x = self.ln_f(x)  # (B, an * T, C)
        x = x.view(bz, (self.n_views + 1) * self.seq_len, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0, 1, 4, 2, 3).contiguous()  # same as token_embeddings

        image_tensor_out = x[:, :self.n_views * self.seq_len, :, :, :].contiguous().view(bz * self.seq_len, -1, h, w)
        lidar_tensor_out = x[:, self.n_views * self.seq_len:, :, :, :].contiguous().view(bz * self.seq_len, -1, h, w)
        image_tensor_out = raw_image_tensor + F.upsample(image_tensor_out,
                                                         size=(raw_image_tensor.shape[2], raw_image_tensor.shape[3]),
                                                         mode='bilinear',
                                                         align_corners=True)

        lidar_tensor_out = raw_lidar_tensor + F.upsample(lidar_tensor_out,
                                                         size=(raw_lidar_tensor.shape[2], raw_lidar_tensor.shape[3]),
                                                         mode='bilinear',
                                                         align_corners=True)

        return image_tensor_out, lidar_tensor_out
