import os
import cv2
import time
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
from dataset.get_loader import get_loader
from config import param as option
from torch.autograd import Variable
from torch.optim import lr_scheduler
from utils import AvgMeter, set_seed, visualize_all
from model.get_model import get_model
from loss.get_loss import get_loss, cal_loss
from optim.get_optim import get_optim, get_optim_dis
from torch.utils.tensorboard import SummaryWriter

if option['task'] == 'Weak-RGB-SOD':
    from trainer.weakly_train import train_one_epoch
elif option['task'] == 'SOD':     
    from trainer.basic_train import train_one_epoch
elif option['task'] == 'RGBD-SOD':
    from trainer.rgbd_train import train_one_epoch


if __name__ == "__main__":
    # Begin the training process
    set_seed(option['seed'])
    loss_fun = get_loss(option)
    model, dis_model = get_model(option)
    optimizer, scheduler = get_optim(option, model.parameters())
    if dis_model is not None:
        optimizer_dis, scheduler_dis = get_optim_dis(option, dis_model.parameters())
    else:
        optimizer_dis, scheduler_dis = None, None
    train_loader = get_loader(option)
    model_list, optimizer_list = [model, dis_model], [optimizer, optimizer_dis]
    writer = SummaryWriter(option['log_path'])
    for epoch in range(1, (option['epoch']+1)):
        model, loss_record = train_one_epoch(epoch, model_list, optimizer_list, train_loader, loss_fun)
        writer.add_scalar('loss', loss_record.show(), epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        scheduler.step()
        if scheduler_dis is not None:
            scheduler_dis.step()

        save_path = option['ckpt_save_path']

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if epoch % option['save_epoch'] == 0:
            torch.save(model.state_dict(), save_path + '/{:d}_{:.4f}'.format(epoch, loss_record.show()) + '.pth')
