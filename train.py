import os
import torch
from glob import glob
from dataset.get_loader import get_loader
from config import param as option
from utils import set_seed, save_scripts
from model.get_model import get_model
from loss.get_loss import get_loss
from optim.get_optim import get_optim, get_optim_dis
from torch.utils.tensorboard import SummaryWriter

# if option['task'] == 'Weak-RGB-SOD':
#     from trainer.weakly_train import train_one_epoch
# elif option['task'] == 'SOD' or option['task'] == 'COD' :     
#     from trainer.diff_basic_train import train_one_epoch
# elif option['task'] == 'RGBD-SOD':
#     from trainer.diff_rgbd_train import train_one_epoch
from trainer.trainer_gan import train_one_epoch

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
    train_loader, dataset_size = get_loader(option)
    model_list, optimizer_list = [model, dis_model], [optimizer, optimizer_dis]
    writer = SummaryWriter(option['log_path'])
    
    save_scripts(option['log_path'], scripts_to_save=glob('*.*'))
    save_scripts(option['log_path'], scripts_to_save=glob('dataset/*.py', recursive=True))
    save_scripts(option['log_path'], scripts_to_save=glob('model/*.py', recursive=True))
    save_scripts(option['log_path'], scripts_to_save=glob('optim/*.py', recursive=True))
    save_scripts(option['log_path'], scripts_to_save=glob('trainer/*.py', recursive=True))
    save_scripts(option['log_path'], scripts_to_save=glob('model/blocks/*.py', recursive=True))
    save_scripts(option['log_path'], scripts_to_save=glob('model/backbone/*.py', recursive=True))
    save_scripts(option['log_path'], scripts_to_save=glob('model/decoder/*.py', recursive=True))
    save_scripts(option['log_path'], scripts_to_save=glob('model/depth_module/*.py', recursive=True))
    save_scripts(option['log_path'], scripts_to_save=glob('model/neck/*.py', recursive=True))

    for epoch in range(1, (option['epoch']+1)):
        model, loss_record = train_one_epoch(epoch, model_list, optimizer_list, train_loader, dataset_size, loss_fun)
        writer.add_scalar('loss', loss_record.show(), epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        scheduler.step()
        if scheduler_dis is not None:
            scheduler_dis.step()

        save_path = option['ckpt_save_path']

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if epoch % option['save_epoch'] == 0:
            torch.save(model.state_dict(), save_path + '/{:0>2d}_{:.4f}'.format(epoch, loss_record.show()) + '.pth')
