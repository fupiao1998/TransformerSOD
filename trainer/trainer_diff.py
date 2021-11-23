import os
import pdb
import cv2
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from config import param as option
from utils import AvgMeter, label_edge_prediction, visualize_list
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
            trainsize = (int(round(option['trainsize'] * rate / 32) * 32), int(round(option['trainsize'] * rate / 32) * 32))
            if rate != 1:
                images = F.upsample(images, size=trainsize, mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=trainsize, mode='bilinear', align_corners=True)
            # Inference Once
            ref_pre = generator(images, depth)
            # Optimize generator
            supervised_loss = cal_loss(ref_pre, gts, loss_fun)
            
            # Optimize discriminator
            dis_loss, diff_loss, Dis_output_list = 0, 0, list()
            if discriminator is not None:
                for pre in ref_pre:
                    dis_pred = pre.detach()
                    output = torch.cat((images, dis_pred), 1)
                    Dis_output = discriminator(output)
                    Dis_output = F.upsample(Dis_output, size=trainsize, mode='bilinear', align_corners=True)
                    Dis_output_list.append(Dis_output.detach())
                    target = gts * (1 - torch.sigmoid(dis_pred)) + (1 - gts) * torch.sigmoid(dis_pred)
                    dis_loss_curr = CE(torch.sigmoid(Dis_output), target.detach())
                    dis_loss = dis_loss + dis_loss_curr

                dis_loss = dis_loss/len(ref_pre)
                dis_loss.backward()
                discriminator_optimizer.step()
            
            for pre, dis_out in zip(ref_pre, Dis_output_list):
                diff_loss_curr = cal_loss(pre, gts, loss_fun, torch.sigmoid(dis_out))
                diff_loss = diff_loss_curr + diff_loss

            diff_loss = diff_loss/len(ref_pre)
            supervised_loss_with_diff = (supervised_loss + diff_loss) / 2
            supervised_loss_with_diff.backward()
            generator_optimizer.step()

            loss = supervised_loss_with_diff + dis_loss

            result_list = [torch.sigmoid(x) for x in ref_pre]
            result_list.append(gts)
            visualize_list(result_list, option['log_path'])

            if rate == 1:
                loss_record.update(loss.data, option['batch_size'])
                supervised_loss_record.update(supervised_loss_with_diff.data, option['batch_size'])
                dis_loss_record.update((loss-supervised_loss_with_diff).data, option['batch_size'])

        progress_bar.set_postfix(loss=f'{loss_record.show():.3f}|{supervised_loss_record.show():.3f}|{dis_loss_record.show():.3f}')

    return generator, loss_record
