import copy
import torch
import torch.nn as nn
from utils import torch_tile, reparametrize
from model.backbone.get_backbone import get_backbone
from model.neck.get_neck import get_neck
from model.decoder.get_decoder import get_decoder
from model.depth_module.get_depth_module import get_depth_module

from model.blocks.base_blocks import FeatureFusionBlock


class sod_model(torch.nn.Module):
    def __init__(self, option):
        super(sod_model, self).__init__()

        self.backbone, self.channel_list = get_backbone(option)
        self.neck = get_neck(option, self.channel_list)
        self.decoder = get_decoder(option)
        self.depth_module = get_depth_module(option)
        self.noise_model = noise_model(option)   # For abp

    def forward(self, img, z=None, gts=None, depth=None):
        if depth is not None:
            if 'head' in self.depth_module.keys():
                img = self.depth_module['head'](img, depth)
            elif 'feature' in self.depth_module.keys():
                depth_features = self.depth_module['feature'](depth)
        
        backbone_features = self.backbone(img)
        neck_features = self.neck(backbone_features)

        if z is not None:
            neck_features = self.noise_model(z, neck_features)

        if depth is not None and 'fusion' in self.depth_module.keys():
            neck_features = self.depth_module['fusion'](neck_features, depth_features)

        outputs = self.decoder(neck_features)

        return outputs


class sod_model_with_vae(torch.nn.Module):
    def __init__(self, option):
        super(sod_model_with_vae, self).__init__()

        self.backbone, self.channel_list = get_backbone(option)
        self.neck = get_neck(option, self.channel_list)
        self.decoder_prior = get_decoder(option)
        self.depth_module = get_depth_module(option)
        self.vae_model = vae_model(option)
        self.decoder_post = copy.deepcopy(self.decoder_prior)

    def forward(self, img, z=None, gts=None, depth=None):
        if depth is not None:
            if 'head' in self.depth_module.keys():
                img = self.depth_module['head'](img, depth)
            elif 'feature' in self.depth_module.keys():
                depth_features = self.depth_module['feature'](depth)
        
        backbone_features = self.backbone(img)
        neck_features = self.neck(backbone_features)

        neck_features_z_prior, neck_features_z_post, kld = self.vae_model(img, neck_features, gts)

        if depth is not None and 'fusion' in self.depth_module.keys():
            neck_features = self.depth_module['fusion'](neck_features, depth_features)

        if gts is not None:   # In the training case with gt
            outputs_prior = self.decoder_prior(neck_features_z_prior)
            outputs_post = self.decoder_post(neck_features_z_post)

            return outputs_prior, outputs_post, kld
        else:   # In the testing case without gt
            outputs = self.decoder_prior(neck_features_z_prior)

            return outputs


class vae_model(nn.Module):
    def __init__(self, option):
        super(vae_model, self).__init__()
        self.enc_x = encode_for_vae(input_channels=3, option=option)
        self.enc_xy = encode_for_vae(input_channels=4, option=option)

        self.noise_model_prior = noise_model(option)
        self.noise_model_post = noise_model(option)

    def forward(self, img, neck_features, y=None):
        if y == None:
            import pdb ; pdb.set_trace()
            mu_prior, logvar_prior, _ = self.enc_x(img)
            z_prior = reparametrize(mu_prior, logvar_prior)
            neck_features = self.noise_model_prior(z_prior, neck_features)

            return neck_features, None, None
        else:
            mu_prior, logvar_prior, dist_prior = self.enc_x(img)
            mu_post, logvar_post, dist_post = self.enc_xy(torch.cat((img, y), 1))
            kld = torch.mean(torch.distributions.kl.kl_divergence(dist_post, dist_prior))
            z_prior = reparametrize(mu_prior, logvar_prior)
            z_post = reparametrize(mu_post, logvar_post)
            neck_features_z_prior = self.noise_model_prior(z_prior, neck_features)
            neck_features_z_post = self.noise_model_post(z_post, neck_features)

            return neck_features_z_prior, neck_features_z_post, kld


class noise_model(nn.Module):
    def __init__(self, option):
        super(noise_model, self).__init__()
        in_channel = option['neck_channel'] + option['latent_dim']
        out_channel = option['neck_channel']
        self.noise_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0)

    def process_z_noise(self, z, feat):
        spatial_axes = [2, 3]
        z_noise = torch.unsqueeze(z, 2)
        z_noise = torch_tile(z_noise, 2, feat.shape[spatial_axes[0]])
        z_noise = torch.unsqueeze(z_noise, 3)
        z_noise = torch_tile(z_noise, 3, feat.shape[spatial_axes[1]])

        return z_noise

    def forward(self, z, neck_features):
        z_noise = self.process_z_noise(z, neck_features[-1])
        neck_feat_with_noise = self.noise_conv(torch.cat((neck_features[-1], z_noise), 1))
        neck_features[-1] = neck_feat_with_noise
        return neck_features


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


class encode_for_vae(nn.Module):
    def __init__(self, input_channels, option):
        super(encode_for_vae, self).__init__()
        channels = option['neck_channel']
        latent_size = option['latent_dim']
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.Conv2d(channels, 2*channels, kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.Conv2d(2*channels, 4*channels, kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(4*channels, 8*channels, kernel_size=4, stride=2, padding=1)
        self.layer5 = nn.Conv2d(8*channels, 8*channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels*2)
        self.bn3 = nn.BatchNorm2d(channels*4)
        self.bn4 = nn.BatchNorm2d(channels*8)
        self.bn5 = nn.BatchNorm2d(channels*8)
        self.channel = channels
        self.hidden_size = option['trainsize'] // 32

        self.fc1 = nn.Linear(channels*8*self.hidden_size*self.hidden_size, latent_size)  # adjust according to input size
        self.fc2 = nn.Linear(channels*8*self.hidden_size*self.hidden_size, latent_size)  # adjust according to input size

        self.leakyrelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.leakyrelu(self.bn1(self.layer1(input)))
        output = self.leakyrelu(self.bn2(self.layer2(output)))
        output = self.leakyrelu(self.bn3(self.layer3(output)))
        output = self.leakyrelu(self.bn4(self.layer4(output)))
        output = self.leakyrelu(self.bn5(self.layer5(output)))
        output = output.view(-1, self.channel * 8 * self.hidden_size * self.hidden_size)  # adjust according to input size
        # output = self.tanh(output)

        mu = self.fc1(output)
        logvar = self.fc2(output)
        dist = torch.distributions.Independent(torch.distributions.Normal(loc=mu, scale=torch.exp(logvar)), 1)

        return mu, logvar, dist


# ##### Temp vae codes (for debug) ####
# class Encode_x(nn.Module):
#     def __init__(self, input_channels, channels, latent_size):
#         super(Encode_x, self).__init__()
#         self.contracting_path = nn.ModuleList()
#         self.input_channels = input_channels
#         self.relu = nn.ReLU(inplace=True)
#         self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
#         self.bn1 = nn.BatchNorm2d(channels)
#         self.layer2 = nn.Conv2d(channels, 2 * channels, kernel_size=4, stride=2, padding=1)
#         self.bn2 = nn.BatchNorm2d(channels * 2)
#         self.layer3 = nn.Conv2d(2 * channels, 4 * channels, kernel_size=4, stride=2, padding=1)
#         self.bn3 = nn.BatchNorm2d(channels * 4)
#         self.layer4 = nn.Conv2d(4 * channels, 8 * channels, kernel_size=4, stride=2, padding=1)
#         self.bn4 = nn.BatchNorm2d(channels * 8)
#         self.layer5 = nn.Conv2d(8 * channels, 8 * channels, kernel_size=4, stride=2, padding=1)
#         self.bn5 = nn.BatchNorm2d(channels * 8)
#         self.channel = channels

#         self.fc1 = nn.Linear(channels * 8 * 12 * 12, latent_size)  # adjust according to input size
#         self.fc2 = nn.Linear(channels * 8 * 12 * 12, latent_size)  # adjust according to input size

#         self.leakyrelu = nn.LeakyReLU()
#         self.tanh = nn.Tanh()

#     def forward(self, input):
#         output = self.leakyrelu(self.bn1(self.layer1(input)))
#         # print(output.size())
#         output = self.leakyrelu(self.bn2(self.layer2(output)))
#         # print(output.size())
#         output = self.leakyrelu(self.bn3(self.layer3(output)))
#         # print(output.size())
#         output = self.leakyrelu(self.bn4(self.layer4(output)))
#         # print(output.size())
#         output = self.leakyrelu(self.bn5(self.layer5(output)))
#         output = output.view(-1, self.channel * 8 * 12 * 12)  # adjust according to input size
#         # print(output.size())
#         # output = self.tanh(output)

#         mu = self.fc1(output)
#         logvar = self.fc2(output)
#         dist = torch.distributions.Independent(torch.distributions.Normal(loc=mu, scale=torch.exp(logvar)), 1)

#         return mu, logvar, dist

# class Encode_xy(nn.Module):
#     def __init__(self, input_channels, channels, latent_size):
#         super(Encode_xy, self).__init__()
#         self.contracting_path = nn.ModuleList()
#         self.input_channels = input_channels
#         self.relu = nn.ReLU(inplace=True)
#         self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
#         self.bn1 = nn.BatchNorm2d(channels)
#         self.layer2 = nn.Conv2d(channels, 2 * channels, kernel_size=4, stride=2, padding=1)
#         self.bn2 = nn.BatchNorm2d(channels * 2)
#         self.layer3 = nn.Conv2d(2 * channels, 4 * channels, kernel_size=4, stride=2, padding=1)
#         self.bn3 = nn.BatchNorm2d(channels * 4)
#         self.layer4 = nn.Conv2d(4 * channels, 8 * channels, kernel_size=4, stride=2, padding=1)
#         self.bn4 = nn.BatchNorm2d(channels * 8)
#         self.layer5 = nn.Conv2d(8 * channels, 8 * channels, kernel_size=4, stride=2, padding=1)
#         self.bn5 = nn.BatchNorm2d(channels * 8)
#         self.channel = channels

#         self.fc1 = nn.Linear(channels * 8 * 12 * 12, latent_size)  # adjust according to input size
#         self.fc2 = nn.Linear(channels * 8 * 12 * 12, latent_size)  # adjust according to input size

#         self.leakyrelu = nn.LeakyReLU()
#         self.tanh = nn.Tanh()

#     def forward(self, input):
#         output = self.leakyrelu(self.bn1(self.layer1(input)))
#         # print(output.size())
#         output = self.leakyrelu(self.bn2(self.layer2(output)))
#         # print(output.size())
#         output = self.leakyrelu(self.bn3(self.layer3(output)))
#         # print(output.size())
#         output = self.leakyrelu(self.bn4(self.layer4(output)))
#         # print(output.size())
#         output = self.leakyrelu(self.bn5(self.layer5(output)))
#         output = output.view(-1, self.channel * 8 * 12 * 12)  # adjust according to input size
#         # print(output.size())
#         # output = self.tanh(output)

#         mu = self.fc1(output)
#         logvar = self.fc2(output)
#         dist = torch.distributions.Independent(torch.distributions.Normal(loc=mu, scale=torch.exp(logvar)), 1)

#         return mu, logvar, dist

# class Interpolate(nn.Module):
#     def __init__(self, scale_factor, mode, align_corners=False):
#         super(Interpolate, self).__init__()

#         self.interp = nn.functional.interpolate
#         self.scale_factor = scale_factor
#         self.mode = mode
#         self.align_corners = align_corners

#     def forward(self, x):
#         x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
#         return x

# class Pred_decoder(nn.Module):
#     # resnet based encoder decoder
#     def __init__(self, channel, latent_dim):
#         super(Pred_decoder, self).__init__()

#         self.path4 = FeatureFusionBlock(channel)
#         self.path3 = FeatureFusionBlock(channel)
#         self.path2 = FeatureFusionBlock(channel)
#         self.path1 = FeatureFusionBlock(channel)

#         self.output_conv = nn.Sequential(
#             nn.Conv2d(channel, channel//2, kernel_size=3, stride=1, padding=1),
#             self.upsample,
#             nn.Conv2d(channel//2, channel//4, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(True),
#             nn.Conv2d(channel//4, 1, kernel_size=1, stride=1, padding=0),
#         )

#         self.noise_conv = nn.Conv2d(channel + latent_dim, channel, kernel_size=1, padding=0)
#         self.spatial_axes = [2, 3]

#     def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
#         return block(dilation_series, padding_series, NoLabels, input_channel)

#     def forward(self, x1, x2, x3, x4, z):
#         conv1_feat = x1
#         conv2_feat = x2
#         conv3_feat = x3
#         conv4_feat = x4

#         z_noise = torch.unsqueeze(z, 2)
#         z_noise = torch_tile(z_noise, 2, conv4_feat.shape[self.spatial_axes[0]])
#         z_noise = torch.unsqueeze(z_noise, 3)
#         z_noise = torch_tile(z_noise, 3, conv4_feat.shape[self.spatial_axes[1]])

#         conv4_feat = torch.cat((conv4_feat, z_noise), 1)
#         conv4_feat = self.noise_conv(conv4_feat)

#         conv4_feat = self.path4(conv4_feat)
#         conv43 = self.path3(conv4_feat, conv3_feat)
#         conv432 = self.path2(conv43, conv2_feat)
#         conv4321 = self.path1(conv432, conv1_feat)

#         pred = self.output_conv(conv4321)

#         return pred

# class sod_model_with_vae_(torch.nn.Module):
#     def __init__(self, option):
#         super(sod_model_with_vae_, self).__init__()

#         self.backbone, self.channel_list = get_backbone(option)
#         self.neck = get_neck(option, self.channel_list)
#         self.relu = nn.ReLU(inplace=True)
#         channel = option['neck_channel']
#         latent_dim = option['latent_dim']

#         self.enc_x = Encode_x(3, channel, latent_dim)
#         self.enc_xy = Encode_xy(4, channel, latent_dim)

#         self.prior_dec = Pred_decoder(channel, latent_dim)
#         self.post_dec = Pred_decoder(channel, latent_dim)

#     def reparametrize(self, mu, logvar):
#         std = logvar.mul(0.5).exp_()
#         eps = torch.cuda.FloatTensor(std.size()).normal_()
#         eps = torch.autograd.Variable(eps)

#         return eps.mul(std).add_(mu)

#     def kl_divergence(self, posterior_latent_space, prior_latent_space):
#         kl_div = torch.distributions.kl.kl_divergence(posterior_latent_space, prior_latent_space)
#         return kl_div

#     def forward(self, img, z=None, gts=None, depth=None):        
#         backbone_features = self.backbone(img)
#         neck_features = self.neck(backbone_features)

#         if gts is not None:   # In the training case with gt
#             mu_prior, logvar_prior, dist_prior = self.enc_x(img)

#             mu_post, logvar_post, dist_post = self.enc_xy(torch.cat((img, gts),1))
#             kld = torch.mean(self.kl_divergence(dist_post, dist_prior))
#             z_prior = self.reparametrize(mu_prior, logvar_prior)
#             z_post = self.reparametrize(mu_post, logvar_post)
#             outputs_prior = self.prior_dec(*neck_features, z_prior)
#             outputs_post = self.post_dec(*neck_features, z_post)
#             return [outputs_prior], [outputs_post], kld
#         else:   # In the testing case without gt
#             import pdb; pdb.set_trace()
#             outputs = self.decoder_prior(neck_features_z_prior)

#             return outputs
# ##### Temp vae codes (for debug) ####