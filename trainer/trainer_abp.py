import os
import pdb
import cv2
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from config import param as option
from utils import AvgMeter, label_edge_prediction, visualize_list
from loss.get_loss import cal_loss
from utils import DotDict
from loss.StructureConsistency import SaliencyStructureConsistency as SSIMLoss


CE = torch.nn.BCELoss()
def train_one_epoch(epoch, model_list, optimizer_list, train_loader, dataset_size, loss_fun):
    ## Setup abp params
    opt = DotDict()
    opt.latent_dim = option['abp_config']['latent_dim']
    opt.langevin_step_num_gen = option['abp_config']['step_num']
    opt.sigma_gen = option['abp_config']['sigma_gen']
    opt.langevin_s = option['abp_config']['langevin_s']
    train_z = torch.FloatTensor(dataset_size, opt.latent_dim).normal_(0, 1).cuda()
    ## Setup abp params

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
            elif len(pack) == 5:
                images, gts, mask, gray, index, depth = pack['image'].cuda(), pack['gt'].cuda(), pack['mask'].cuda(), pack['gray'].cuda(), pack['index'], None

            # multi-scale training samples
            trainsize = (int(round(option['trainsize'] * rate / 32) * 32), int(round(option['trainsize'] * rate / 32) * 32))
            if rate != 1:
                images = F.upsample(images, size=trainsize, mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=trainsize, mode='bilinear', align_corners=True)

            z_noise = train_z[index]
            z_noise = Variable(z_noise, requires_grad=True)
            z_noise_preds = [z_noise.clone() for _ in range(opt.langevin_step_num_gen + 1)]
            for kk in range(opt.langevin_step_num_gen):
                z_noise = Variable(z_noise_preds[kk], requires_grad=True).cuda()
                noise = torch.randn(z_noise.size()).cuda()

                gen_res = generator(img=images, z=z_noise, depth=depth)['sal_pre']
                gen_loss = 0
                for i in gen_res:
                    if option['task'].lower() == 'weak-rgb-sod':
                        gen_loss += 1 / (2.0 * opt.sigma_gen * opt.sigma_gen) * F.mse_loss(torch.sigmoid(i)*mask, gts*mask, size_average=True, reduction='sum')
                    else:
                        gen_loss += 1 / (2.0 * opt.sigma_gen * opt.sigma_gen) * F.mse_loss(torch.sigmoid(i), gts, size_average=True, reduction='sum')
                gen_loss.backward(torch.ones(gen_loss.size()).cuda())

                grad = z_noise.grad
                z_noise = z_noise + 0.5 * opt.langevin_s * opt.langevin_s * grad
                z_noise += opt.langevin_s * noise
                z_noise_preds[kk + 1] = z_noise

            z_noise_ref = z_noise
            
            pred = generator(img=images, z=z_noise_ref, depth=depth)
            sal_pred = pred['sal_pre']

            ## Caltulate loss
            # loss_init, loss_ref = cal_loss(pred_init, gts, loss_fun), cal_loss(pred_ref, gts, loss_fun)
            if option['task'].lower() == 'sod':
                supervised_loss = cal_loss(sal_pred, gts, loss_fun)
            elif option['task'].lower() == 'weak-rgb-sod':
                supervised_loss = loss_fun(images=images, outputs=sal_pred, gt=gts, masks=mask, grays=gray, model=generator)

            supervised_loss.backward()
            generator_optimizer.step()

            result_list = [torch.sigmoid(x) for x in sal_pred]
            result_list.append(gts)
            visualize_list(result_list, option['log_path'])

            if rate == 1:
                loss_record.update(supervised_loss.data, option['batch_size'])
                supervised_loss_record.update(supervised_loss.data, option['batch_size'])
                dis_loss_record.update(supervised_loss.data, option['batch_size'])

        progress_bar.set_postfix(loss=f'{loss_record.show():.3f}|{supervised_loss_record.show():.3f}|{dis_loss_record.show():.3f}')

    return {'generator': generator, "discriminator": discriminator}, loss_record
