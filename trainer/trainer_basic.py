import os
import pdb
import cv2
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from config import param as option
from utils import AvgMeter, label_edge_prediction, visualize_list, make_dis_label
from loss.get_loss import cal_loss


CE = torch.nn.BCELoss()
def train_one_epoch(epoch, model_list, optimizer_list, train_loader, dataset_size, loss_fun):

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

            pred = generator(img=images)
            loss_all = cal_loss(pred, gts, loss_fun)

            loss_all.backward()
            generator_optimizer.step()

            result_list = [torch.sigmoid(x) for x in pred]
            result_list.append(gts)
            visualize_list(result_list, option['log_path'])
            del result_list

            if rate == 1:
                loss_record.update(loss_all.data, option['batch_size'])

        progress_bar.set_postfix(loss=f'{loss_record.show():.3f}|{loss_record.show():.3f}|{loss_record.show():.3f}')

    return {'generator': generator, "discriminator": discriminator}, loss_record
