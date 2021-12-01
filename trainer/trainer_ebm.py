import os
import pdb
import cv2
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from config import param as option
from utils import AvgMeter, visualize_list, make_dis_label, sample_p_0, compute_energy
from loss.get_loss import cal_loss
from utils import DotDict


CE = torch.nn.BCELoss()
def train_one_epoch(epoch, model_list, optimizer_list, train_loader, dataset_size, loss_fun):
    ## Setup ebm params
    opt = DotDict()
    opt.ebm_out_dim = 1
    opt.ebm_middle_dim = 100
    opt.latent_dim = 32
    opt.e_init_sig = 1.0
    opt.e_l_steps = 5
    opt.e_l_step_size = 0.4
    opt.e_prior_sig = 1.0
    opt.g_l_steps = 5
    opt.g_llhd_sigma = 0.3
    opt.g_l_step_size = 0.1
    opt.e_energy_form = 'identity'
    ## Setup ebm params

    generator, ebm_model = model_list
    generator_optimizer, ebm_model_optimizer = optimizer_list
    generator.train()
    if ebm_model is not None:
        ebm_model.train()
    loss_record, supervised_loss_record, dis_loss_record = AvgMeter(), AvgMeter(), AvgMeter()
    print('Learning Rate: {:.2e}'.format(generator_optimizer.param_groups[0]['lr']))
    progress_bar = tqdm(train_loader, desc='Epoch[{:03d}/{:03d}]'.format(epoch, option['epoch']))
    for i, pack in enumerate(progress_bar):
        for rate in option['size_rates']:
            generator_optimizer.zero_grad()
            if ebm_model is not None:
                ebm_model_optimizer.zero_grad()
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

            z_e_0 = sample_p_0(images, opt)
            z_g_0 = sample_p_0(images, opt)

            z_e_0 = Variable(z_e_0)
            z = z_e_0.clone().detach()
            z.requires_grad = True
            for kk in range(opt.e_l_steps):
                en = ebm_model(z)
                z_grad = torch.autograd.grad(en.sum(), z)[0]
                z.data = z.data - 0.5 * opt.e_l_step_size * opt.e_l_step_size * (
                        z_grad + 1.0 / (opt.e_prior_sig * opt.e_prior_sig) * z.data)
                z.data += opt.e_l_step_size * torch.randn_like(z).data
            z_e_noise = z.detach()  ## z_

            z_g_0 = Variable(z_g_0)
            z = z_g_0.clone().detach()
            z.requires_grad = True
            for kk in range(opt.g_l_steps):
                gen_res = generator(images, z)
                g_log_lkhd = 1.0 / (2.0 * opt.g_llhd_sigma * opt.g_llhd_sigma) * F.mse_loss(
                    torch.sigmoid(gen_res[0]), gts)
                z_grad_g = torch.autograd.grad(g_log_lkhd, z)[0]

                en = ebm_model(z)
                z_grad_e = torch.autograd.grad(en.sum(), z)[0]

                z.data = z.data - 0.5 * opt.g_l_step_size * opt.g_l_step_size * (
                        z_grad_g + z_grad_e + 1.0 / (opt.e_prior_sig * opt.e_prior_sig) * z.data)
                z.data += opt.g_l_step_size * torch.randn_like(z).data

            z_g_noise = z.detach()  ## z+

            pred = generator(img=images, z=z_g_noise, depth=depth)
            loss_all = cal_loss(pred, gts, loss_fun)

            loss_all.backward()
            generator_optimizer.step()

            ## learn the ebm
            en_neg = compute_energy(option=opt, score=ebm_model(z_e_noise.detach())).mean()
            en_pos = compute_energy(option=opt, score=ebm_model(z_g_noise.detach())).mean()
            loss_e = en_pos - en_neg
            loss_e.backward()
            ebm_model_optimizer.step()

            result_list = [torch.sigmoid(x) for x in pred]
            result_list.append(gts)
            visualize_list(result_list, option['log_path'])

            if rate == 1:
                loss_record.update(loss_all.data, option['batch_size'])
                supervised_loss_record.update(en_pos.data, option['batch_size'])
                dis_loss_record.update(en_neg.data, option['batch_size'])

        progress_bar.set_postfix(loss=f'{loss_record.show():.3f}|{supervised_loss_record.show():.3f}|{dis_loss_record.show():.3f}')

    return {'generator': generator, "ebm_model": ebm_model}, loss_record
