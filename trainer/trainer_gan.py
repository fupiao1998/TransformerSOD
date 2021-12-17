import os
import pdb
import cv2
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from config import param as option
from utils import AvgMeter, label_edge_prediction, visualize_list, make_dis_label
from loss.get_loss import cal_loss
from utils import DotDict
from loss.StructureConsistency import SaliencyStructureConsistency as SSIMLoss


CE = torch.nn.BCELoss()
def train_one_epoch(epoch, model_list, optimizer_list, train_loader, dataset_size, loss_fun):
    ## Setup gan params
    opt = DotDict()
    opt.latent_dim = option['gan_config']['latent_dim']
    opt.pred_label = option['gan_config']['pred_label']
    opt.gt_label = option['gan_config']['gt_label']
    ## Setup gan params

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
                images, gts, mask, gray, depth = pack['image'].cuda(), pack['gt'].cuda(), pack['mask'].cuda(), pack['gray'].cuda(), None

            # multi-scale training samples
            trainsize = (int(round(option['trainsize']*rate/32)*32), int(round(option['trainsize']*rate/32)*32))
            if rate != 1:
                images = F.upsample(images, size=trainsize, mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=trainsize, mode='bilinear', align_corners=True)

            z_noise = torch.randn(images.shape[0], opt.latent_dim).cuda()
            pred = generator(img=images, z=z_noise, depth=depth)
            sal_pred = pred['sal_pre']
            if option['task'].lower() == 'sod':
                Dis_output = discriminator(torch.cat((images, torch.sigmoid(sal_pred[0]).detach()), 1))
            elif option['task'].lower() == 'weak-rgb-sod':
                Dis_output = discriminator(torch.cat((images, mask*torch.sigmoid(sal_pred[0]).detach()), 1))

            up_size = (images.shape[2], images.shape[3])
            Dis_output = F.upsample(Dis_output, size=up_size, mode='bilinear', align_corners=True)
            
            loss_dis_output = CE(torch.sigmoid(Dis_output), make_dis_label(opt.gt_label, gts))
            
            if option['task'].lower() == 'sod':
                import pdb; pdb.set_trace()
                supervised_loss = cal_loss(pred['sal_pre'], gts, loss_fun)
            elif option['task'].lower() == 'weak-rgb-sod':
                supervised_loss = loss_fun(images=images, outputs=pred['sal_pre'], gt=gts, masks=mask, grays=gray, model=generator)

            loss_all = supervised_loss + 0.1*loss_dis_output

            loss_all.backward()
            generator_optimizer.step()

            # train discriminator
            dis_pred = torch.sigmoid(sal_pred[0]).detach()
            if option['task'].lower() == 'sod':
                Dis_output = discriminator(torch.cat((images, dis_pred), 1))
            elif option['task'].lower() == 'weak-rgb-sod':
                Dis_output = discriminator(torch.cat((images, mask*dis_pred), 1))
            Dis_target = discriminator(torch.cat((images, gts), 1))
            Dis_output = F.upsample(torch.sigmoid(Dis_output), size=up_size, mode='bilinear', align_corners=True)
            Dis_target = F.upsample(torch.sigmoid(Dis_target), size=up_size, mode='bilinear', align_corners=True)

            loss_dis_output = CE(torch.sigmoid(Dis_output), make_dis_label(opt.pred_label, gts))
            loss_dis_target = CE(torch.sigmoid(Dis_target), make_dis_label(opt.gt_label, gts))
            dis_loss = 0.5 * (loss_dis_output + loss_dis_target)
            dis_loss.backward()
            discriminator_optimizer.step()

            result_list = [torch.sigmoid(x) for x in sal_pred]
            result_list.append(gts)
            visualize_list(result_list, option['log_path'])

            if rate == 1:
                loss_record.update(supervised_loss.data, option['batch_size'])
                supervised_loss_record.update((loss_all-supervised_loss).data, option['batch_size'])
                dis_loss_record.update(dis_loss.data, option['batch_size'])

        progress_bar.set_postfix(loss=f'{loss_record.show():.3f}|{supervised_loss_record.show():.3f}|{dis_loss_record.show():.3f}')

    return {'generator': generator, "discriminator": discriminator}, loss_record
