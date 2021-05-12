from loss.structure_loss import structure_loss


def get_loss(option):
    task = option['task']
    # if task == 'COD' or task == 'SOD':
    loss_fun = structure_loss

    return loss_fun


def cal_loss(pred, gt, loss_fun, weight=None):
    if isinstance(pred, list):
        loss = 0
        for i in pred:
            loss_curr = loss_fun(i, gt, weight)
            loss += loss_curr
        loss = loss / len(pred)
    else:
        loss = loss_fun(pred, gt, weight)

    return loss
