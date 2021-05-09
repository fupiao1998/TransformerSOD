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
def train_one_epoch(epoch, model, generator_optimizer, train_loader, loss_fun):
    model.train()
    loss_record, loss_scale_record = AvgMeter(), AvgMeter()
    print('Learning Rate: {:.2e}'.format(generator_optimizer.param_groups[0]['lr']))
    progress_bar = tqdm(train_loader, desc='Epoch[{:03d}/{:03d}]'.format(epoch, option['epoch']))
    for i, pack in enumerate(progress_bar):
        for rate in option['size_rates']:
            generator_optimizer.zero_grad()
            images, gts, masks, grays = pack[0].cuda(), pack[1].cuda(), pack[2].cuda(), pack[3].cuda()

            # Inference Once
            ref_pre, edge_map = model(images)
            img_size = images.size(2) * images.size(3) * images.size(0)
            ratio = img_size / torch.sum(masks)

            images_trans, ref_pre_trans = rot_trans(images, ref_pre)
            images_trans_pre, _ = model(images_trans)
            cycle_loss = SaliencyStructureConsistency(torch.sigmoid(ref_pre_trans[0]), torch.sigmoid(images_trans_pre[0]), alpha=0.85)

            sample = {'rgb': images.clone()}
            loss_lsc_sal = loss_lsc(torch.sigmoid(ref_pre[0]), loss_lsc_kernels_desc_defaults, loss_lsc_radius, 
                                    sample, images.shape[2], images.shape[3])['loss']

            sal_prob = torch.sigmoid(ref_pre[0])
            sal_prob = sal_prob * masks
            smoothLoss_cur = 0.3 * smooth_loss(torch.sigmoid(ref_pre[0]), grays)
            sal_loss = ratio * CE(sal_prob, gts * masks) + smoothLoss_cur + weight_lsc*loss_lsc_sal

            edges_gt = torch.sigmoid(ref_pre[0]).detach()
            edge_loss = 1.0 * CE(torch.sigmoid(edge_map), label_edge_prediction(edges_gt))
            loss = edge_loss + sal_loss + cycle_loss

            loss.backward()
            generator_optimizer.step()
            visualize_list([torch.sigmoid(ref_pre[0]), gts, torch.sigmoid(edge_map), label_edge_prediction(edges_gt)], option['log_path'])

            if rate == 1:
                loss_record.update(loss.data, option['batch_size'])
                # loss_scale_record.update(loss_scale.data, option['batch_size'])

        progress_bar.set_postfix(loss=f'{loss_record.show():.5f}')

    return model, loss_record
