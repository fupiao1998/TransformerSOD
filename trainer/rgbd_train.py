import os
import cv2
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from config import param as option
from torch.autograd import Variable
from utils import AvgMeter, visualize_all, label_edge_prediction, visualize_list
from loss.get_loss import get_loss, cal_loss
from loss.StructureConsistency import SaliencyStructureConsistency
from img_trans import rot_trans, scale_trans


CE = torch.nn.BCELoss()
def train_one_epoch(epoch, model_list, optimizer_list, train_loader, loss_fun):
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
            images, gts, depths = pack[0].cuda(), pack[1].cuda(), pack[2].cuda()

            # multi-scale training samples
            trainsize = int(round(option['trainsize'] * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                depths = F.upsample(depths, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # Inference Once
            ref_pre = generator(images, depths)
            # Optimize generator
            supervised_loss = cal_loss(ref_pre, gts, loss_fun)
            supervised_loss.backward()
            generator_optimizer.step()
            # Optimize discriminator
            dis_loss = 0
            if discriminator is not None:
                for pre in ref_pre:
                    dis_pred = pre.detach()
                    output = torch.cat((images, dis_pred), 1)
                    Dis_output = discriminator(output)
                    Dis_output = F.upsample(Dis_output, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    target = gts * (1 - torch.sigmoid(dis_pred)) + (1 - gts) * torch.sigmoid(dis_pred)
                    dis_loss_curr = CE(torch.sigmoid(Dis_output), target.detach())
                    dis_loss = dis_loss + dis_loss_curr

                dis_loss = dis_loss/len(ref_pre)
                dis_loss.backward()
                discriminator_optimizer.step()

            loss = supervised_loss + dis_loss

            visualize_list([torch.sigmoid(ref_pre[0]), torch.sigmoid(ref_pre[1]), torch.sigmoid(ref_pre[2]), torch.sigmoid(ref_pre[3]), gts], option['log_path'])

            if rate == 1:
                loss_record.update(loss.data, option['batch_size'])
                supervised_loss_record.update(supervised_loss.data, option['batch_size'])
                dis_loss_record.update((loss-supervised_loss).data, option['batch_size'])

        progress_bar.set_postfix(loss=f'{loss_record.show():.3f}|{supervised_loss_record.show():.3f}|{dis_loss_record.show():.3f}')

    return generator, loss_record
