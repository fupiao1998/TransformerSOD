import os
import cv2
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from config import param as option
from torch.autograd import Variable
from utils import AvgMeter, visualize_all, label_edge_prediction, visualize_list
from loss.lscloss import *
import loss.smoothness
from loss.StructureConsistency import SaliencyStructureConsistency
from img_trans import rot_trans, scale_trans

torch.autograd.set_detect_anomaly(True)
smooth_loss = loss.smoothness.smoothness_loss(size_average=True)
loss_lsc = LocalSaliencyCoherence().cuda()

loss_lsc_kernels_desc_defaults = [{"weight": 1, "xy": 6, "rgb": 0.1}]
loss_lsc_radius = 5
weight_lsc = 0.3
CE = torch.nn.BCELoss()
def train_one_epoch(epoch, model_list, optimizer_list, train_loader, loss_fun):
    generator, discriminator = model_list
    generator_optimizer, discriminator_optimizer = optimizer_list
    generator.train()
    if discriminator is not None:
        discriminator.train()
    loss_record, dis_loss_record = AvgMeter(), AvgMeter()
    print('Learning Rate: {:.2e}'.format(generator_optimizer.param_groups[0]['lr']))
    progress_bar = tqdm(train_loader, desc='Epoch[{:03d}/{:03d}]'.format(epoch, option['epoch']))
    trainsize = int(round(option['trainsize'] * 1 / 32) * 32)
    for i, pack in enumerate(progress_bar):
        for rate in option['size_rates']:
            generator_optimizer.zero_grad()
            if discriminator is not None:
                discriminator_optimizer.zero_grad()
            images, gts, masks, grays = pack[0].cuda(), pack[1].cuda(), pack[2].cuda(), pack[3].cuda()

            ### train generator
            ref_pre = generator(images)
            img_size = images.size(2) * images.size(3) * images.size(0)
            ratio = img_size / torch.sum(masks)
            sample = {'rgb': images.clone()}
            loss_lsc_sal, sal_loss = 0, 0
            for pre in ref_pre:
                loss_lsc_curr = loss_lsc(torch.sigmoid(pre), loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, images.shape[2], images.shape[3])['loss']
                loss_lsc_sal = loss_lsc_sal + loss_lsc_curr

            for pre in ref_pre:
                sal_prob = torch.sigmoid(pre) * masks
                smoothLoss = 0.3 * smooth_loss(torch.sigmoid(pre), grays)
                sal_loss = sal_loss + ratio * CE(sal_prob, gts * masks) + smoothLoss
            
            images_trans, sal_list_trans = scale_trans(images, [ref_pre[-1]])
            ref_trans_pre = generator(images_trans)
            # import pdb; pdb.set_trace()
            cycle_loss = SaliencyStructureConsistency(torch.sigmoid(ref_trans_pre[-1]), torch.sigmoid(sal_list_trans[-1]))

            supervised_loss = (weight_lsc * loss_lsc_sal + sal_loss) / len(ref_pre) + cycle_loss
            
            ### train discriminator
            if discriminator is not None:
                dis_loss, diff_loss, Dis_output_list = 0, 0, list()
                for pre in ref_pre:
                    dis_pred = pre.detach()
                    output = torch.cat((images, dis_pred), 1)
                    Dis_output = discriminator(output)
                    Dis_output = F.upsample(Dis_output, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    Dis_output_list.append(Dis_output)
                    target = masks*(gts * (1 - torch.sigmoid(dis_pred)) + (1 - gts) * torch.sigmoid(dis_pred))
                    dis_loss_curr = CE(torch.sigmoid(Dis_output)*masks, target.detach())
                    dis_loss = dis_loss + dis_loss_curr
                dis_loss = dis_loss / len(ref_pre)

                dis_loss.backward()
                discriminator_optimizer.step()
            
            for pre, dis_out in zip(ref_pre, Dis_output_list):
                diff_loss_curr = cal_loss(pre, gts, loss_fun, torch.sigmoid(dis_out))
                diff_loss = diff_loss_curr + diff_loss
            diff_loss = diff_loss/len(ref_pre)

            supervised_loss.backward()
            generator_optimizer.step()

            visualize_list([torch.sigmoid(ref_pre[0]), torch.sigmoid(ref_pre[1]), torch.sigmoid(ref_pre[2]), torch.sigmoid(ref_pre[3], torch.sigmoid(ref_pre[4]), gts], option['log_path'])

            if rate == 1:
                loss_record.update(loss.data, option['batch_size'])
                dis_loss_record.update(loss.data, option['batch_size'])
                # loss_scale_record.update(loss_scale.data, option['batch_size'])

        progress_bar.set_postfix(loss=f'{loss_record.show():.3f}|{dis_loss_record.show():.3f}')

    return generator, loss_record
