import torch
from torch.optim import lr_scheduler


def get_optim(option, params):
    optimizer = getattr(torch.optim, option['optim'])(params, option['lr_config']['lr'], betas=option['lr_config']['beta'])
    # optimizer = torch.optim.SGD(params, 1e-3, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=option['lr_config']['decay_epoch'], gamma=option['lr_config']['decay_rate'])
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=option['lr_config']['gamma'])

    return optimizer, scheduler


def get_optim_dis(option, params):
    optimizer = getattr(torch.optim, option['optim'])(params, option['lr_config']['lr_dis'], betas=option['lr_config']['beta'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=option['lr_config']['decay_epoch'], gamma=option['lr_config']['decay_rate'])

    return optimizer, scheduler
