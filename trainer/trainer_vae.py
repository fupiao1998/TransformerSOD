import os
import pdb
import cv2
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from config import param as option
from utils import AvgMeter, label_edge_prediction, visualize_list, l2_regularisation, linear_annealing
from loss.get_loss import cal_loss
from utils import DotDict


CE = torch.nn.BCELoss()
mse_loss = torch.nn.MSELoss(size_average=True, reduction='sum')
def train_one_epoch(epoch, model_list, optimizer_list, train_loader, dataset_size, loss_fun):
    ## Setup vae params
    opt = DotDict()
    opt.reg_weight = option['vae_config']['reg_weight']
    opt.lat_weight = option['vae_config']['lat_weight']
    opt.vae_loss_weight = option['vae_config']['vae_loss_weight']
    ## Setup vae params

    generator, discriminator = model_list
    generator_optimizer, discriminator_optimizer = optimizer_list
    generator.train()
    if discriminator is not None:
        discriminator.train()
    loss_record, supervised_loss_record, dis_loss_record = AvgMeter(), AvgMeter(), AvgMeter()
    print('Learning Rate: {:.2e}'.format(generator_optimizer.param_groups[0]['lr']))
    progress_bar = tqdm(train_loader, desc='Epoch[{:03d}/{:03d}]'.format(epoch, option['epoch']))
    for i, pack in enumerate(progress_bar):
        for rate in option['size_rates']:
            generator_optimizer.zero_grad()
            if discriminator is not None:
                discriminator_optimizer.zero_grad()
            if len(pack) == 3:
                images, gts, depth, index = pack['image'].cuda(), pack['gt'].cuda(), None, pack['index']
            elif len(pack) == 4:
                images, gts, depth, index = pack['image'].cuda(), pack['gt'].cuda(), pack['depth'].cuda(), pack['index']
            # elif len(pack) == 4:
            #     images, gts, mask, gray = pack['image'].cuda(), pack['gt'].cuda(), pack['mask'].cuda(), pack['gray'].cuda()

            # multi-scale training samples
            trainsize = (int(round(option['trainsize']*rate/32)*32), int(round(option['trainsize']*rate/32)*32))
            if rate != 1:
                images = F.upsample(images, size=trainsize, mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=trainsize, mode='bilinear', align_corners=True)

            pred_prior, pred_post, latent_loss = generator(img=images, gts=gts)
            reg_loss = l2_regularisation(generator.vae_model.enc_x) + \
                       l2_regularisation(generator.vae_model.enc_xy) + \
                       l2_regularisation(generator.decoder_prior) + \
                       l2_regularisation(generator.decoder_post)
            reg_loss = opt.reg_weight * reg_loss
            anneal_reg = 0.01  # linear_annealing(0, 1, epoch, option['epoch'])
            loss_latent = opt.lat_weight * anneal_reg * latent_loss
            gen_loss_cvae = opt.vae_loss_weight * (cal_loss(pred_post, gts, loss_fun) + loss_latent)  # BUG: Only support for single out
            gen_loss_gsnn = (1 - opt.vae_loss_weight) * cal_loss(pred_prior, gts, loss_fun)  # BUG: Only support for single out
            loss_all = gen_loss_cvae + gen_loss_gsnn + reg_loss
            loss_all.backward()
            generator_optimizer.step()

            result_list = [torch.sigmoid(x) for x in pred_prior]
            result_list.append(gts)
            visualize_list(result_list, option['log_path'])

            if rate == 1:
                loss_record.update(gen_loss_cvae.data, option['batch_size'])
                supervised_loss_record.update(gen_loss_gsnn.data, option['batch_size'])
                dis_loss_record.update(reg_loss.data, option['batch_size'])

        progress_bar.set_postfix(loss=f'{loss_record.show():.3f}|{supervised_loss_record.show():.3f}|{dis_loss_record.show():.3f}')

    return {'generator': generator, "discriminator": discriminator}, loss_record
