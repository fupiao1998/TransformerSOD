import os
import cv2
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from config import param as option
from torch.autograd import Variable
from utils import AvgMeter, visualize_all, label_edge_prediction
from loss.get_loss import get_loss, cal_loss
from loss.StructureConsistency import SaliencyStructureConsistency
from img_trans import rot_trans, scale_trans


CE = torch.nn.BCELoss()
def train_one_epoch(epoch, model, generator_optimizer, train_loader, loss_fun):
    model.train()
    loss_record, loss_scale_record = AvgMeter(), AvgMeter()
    print('Learning Rate: {:.2e}'.format(generator_optimizer.param_groups[0]['lr']))
    progress_bar = tqdm(train_loader, desc='Epoch[{:03d}/{:03d}]'.format(epoch, option['epoch']))
    for i, pack in enumerate(progress_bar):
        for rate in option['size_rates']:
            generator_optimizer.zero_grad()
            images, gts = pack[0].cuda(), pack[1].cuda()

            # multi-scale training samples
            trainsize = int(round(option['trainsize'] * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # Inference Once
            ref_pre, edge_map = model(images)
            edge_map = F.upsample(edge_map, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            edges_gt = torch.sigmoid(ref_pre[0]).detach()
            edge_loss = CE(torch.sigmoid(edge_map), label_edge_prediction(edges_gt))
            supervised_loss = cal_loss(ref_pre, gts, loss_fun)

            # Inference Twice if Necessary
            if option['scale_trans_radio'] > 0:
                images_trans, ref_pre_trans = scale_trans(images, ref_pre)
                images_trans_pre = model(images_trans)
                cycle_loss = SaliencyStructureConsistency(torch.sigmoid(ref_pre_trans[0]), torch.sigmoid(images_trans_pre[0]))
                loss = supervised_loss + option['scale_trans_radio']*cycle_loss + edge_loss
            elif option['rot_trans_radio'] > 0:
                images_trans, ref_pre_trans = rot_trans(images, ref_pre)
                images_trans_pre = model(images_trans)
                cycle_loss = SaliencyStructureConsistency(torch.sigmoid(ref_pre_trans[0]), torch.sigmoid(images_trans_pre[0]))
                loss = supervised_loss + option['scale_trans_radio']*cycle_loss + edge_loss
            else:
                loss = supervised_loss + edge_loss

            loss.backward()
            generator_optimizer.step()
            visualize_all(torch.sigmoid(ref_pre[0]), gts, option['log_path'])

            if rate == 1:
                loss_record.update(loss.data, option['batch_size'])
                # loss_scale_record.update(loss_scale.data, option['batch_size'])

        progress_bar.set_postfix(loss=f'{loss_record.show():.5f}')

    return model, loss_record
