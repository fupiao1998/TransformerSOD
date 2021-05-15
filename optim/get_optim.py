import torch
from torch.optim import lr_scheduler


def get_optim(option, params):
    optimizer = getattr(torch.optim, option['optim'])(params, option['lr'], betas=option['beta'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=option['decay_epoch'], gamma=option['decay_rate'])

    return optimizer, scheduler


def get_optim_dis(option, params):
    optimizer = getattr(torch.optim, option['optim'])(params, option['lr_dis'], betas=option['beta'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=option['decay_epoch'], gamma=option['decay_rate'])

    return optimizer, scheduler
