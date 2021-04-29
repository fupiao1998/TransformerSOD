import os
import cv2
import time
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
from data import get_loader
from img_trans import scale_trans
from config import param as option
from torch.autograd import Variable
from torch.optim import lr_scheduler
from utils import AvgMeter, set_seed, visualize_all
from model.get_model import get_model
from loss.get_loss import get_loss, cal_loss
from loss.StructureConsistency import SaliencyStructureConsistency
from img_trans import rot_trans, scale_trans
from torch.utils.tensorboard import SummaryWriter


def get_optim(option, params):
    optimizer = torch.optim.Adam(params, option['lr'], betas=option['beta'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=option['decay_epoch'], gamma=option['decay_rate'])

    return optimizer, scheduler


def train_one_epoch(model, generator_optimizer, train_loader, loss_fun):
    model.train()
    loss_record, loss_scale_record = AvgMeter(), AvgMeter()
    print('Learning Rate: {:.2e}'.format(generator_optimizer.param_groups[0]['lr']))
    progress_bar = tqdm(train_loader, desc='Epoch[{:03d}/{:03d}]'.format(epoch, option['epoch']))
    for i, pack in enumerate(progress_bar):
        for rate in size_rates:
            generator_optimizer.zero_grad()
            images, gts = pack[0].cuda(), pack[1].cuda()

            # multi-scale training samples
            trainsize = int(round(option['trainsize'] * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            ref_pre = model(images)
            # images_trans, ref_pre_trans = scale_trans(images, ref_pre)
            # images_trans_pre = model(images_trans)
            # import pdb; pdb.set_trace()
            loss = cal_loss(ref_pre, gts, loss_fun) # + 0.5*SaliencyStructureConsistency(torch.sigmoid(ref_pre_trans[0]), torch.sigmoid(images_trans_pre[0]))
            loss.backward()
            generator_optimizer.step()
            visualize_all(torch.sigmoid(ref_pre[0]), gts, option['log_path'])

            if rate == 1:
                loss_record.update(loss.data, option['batch_size'])
                # loss_scale_record.update(loss_scale.data, option['batch_size'])

        progress_bar.set_postfix(loss=f'{loss_record.show():.5f}')

    return model, loss_record


if __name__ == "__main__":
    # Begin the training process
    set_seed(option['seed'])
    loss_fun = get_loss(option)
    model = get_model(option)
    optimizer, scheduler = get_optim(option, model.parameters())
    train_loader = get_loader(image_root=option['image_root'], gt_root=option['gt_root'], 
                              batchsize=option['batch_size'], trainsize=option['trainsize'])

    size_rates = option['size_rates']  # multi-scale training
    writer = SummaryWriter(option['log_path'])
    for epoch in range(1, (option['epoch']+1)):
        model, loss_record = train_one_epoch(model, optimizer, train_loader, loss_fun)
        writer.add_scalar('loss', loss_record.show(), epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        scheduler.step()

        save_path = option['ckpt_save_path']

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if epoch % option['save_epoch'] == 0:
            torch.save(model.state_dict(), save_path + '/{:d}_{:.4f}'.format(epoch, loss_record.show()) + '.pth')
