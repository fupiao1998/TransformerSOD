import torch
from loss.structure_loss import structure_loss



def bce_loss_with_sigmoid(pred, gt, weight=None):    
    return torch.nn.functional.binary_cross_entropy(torch.sigmoid(pred), gt)


def get_loss(option):
    # task = option['task']
    # if task == 'COD' or task == 'SOD':
    if option['loss'] == 'structure':
        loss_fun = structure_loss
    elif option['loss'] == 'bce':
        loss_fun = bce_loss_with_sigmoid

    return loss_fun


def cal_loss(pred, gt, loss_fun, weight=None):
    if isinstance(pred, list):
        loss = 0
        for i in pred:
            loss_curr = loss_fun(i, gt, weight)
            loss += loss_curr
        loss = loss / len(pred)
    if isinstance(pred, dict):
        for key in pred.keys():
            loss_curr = loss_fun(pred(key), gt, weight)
            loss += loss_curr
        loss = loss / len(pred)
    else:
        loss = loss_fun(pred, gt, weight)

    return loss
