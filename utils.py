import os
import cv2
import shutil
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np
import random


def label_edge_prediction(label):
    fx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)
    fy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).astype(np.float32)
    fx = np.reshape(fx, (1, 1, 3, 3))
    fy = np.reshape(fy, (1, 1, 3, 3))
    fx = (torch.from_numpy(fx)).cuda()
    fy = (torch.from_numpy(fy)).cuda()
    contour_th = 1.5
    # convert label to edge
    label = label.gt(0.5).float()
    label = F.pad(label, (1, 1, 1, 1), mode='replicate')
    label_fx = F.conv2d(label, fx)
    label_fy = F.conv2d(label, fy)
    label_grad = torch.sqrt(torch.mul(label_fx, label_fx) + torch.mul(label_fy, label_fy))
    label_grad = torch.gt(label_grad, contour_th).float()

    return label_grad


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        a = len(self.losses)
        b = np.maximum(a-self.num, 0)
        c = self.losses[b:]

        return torch.mean(torch.stack(c))


def set_seed(seed=1024):
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def visualize_all(in_pred, in_gt, path):
    for kk in range(in_pred.shape[0]):
        pred, gt = in_pred[kk, :, :, :], in_gt[kk, :, :, :]
        pred = (pred.detach().cpu().numpy().squeeze()*255.0).astype(np.uint8)
        gt = (gt.detach().cpu().numpy().squeeze()*255.0).astype(np.uint8)
        cat_img = cv2.hconcat([pred, gt])
        save_path = path + '/vis_temp/'   
        # Save vis images this temp folder, based on this experiment's folder.
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        name = '{:02d}_cat.png'.format(kk)
        cv2.imwrite(save_path + name, cat_img)


def visualize_list(input_list, path):
    for kk in range(input_list[0].shape[0]):
        show_list = []
        for i in input_list:
            tmp = i[kk, :, :, :]
            tmp = (tmp.detach().cpu().numpy().squeeze()*255.0).astype(np.uint8)
            show_list.append(tmp)
        cat_img = cv2.hconcat(show_list)
        save_path = path + '/vis_temp/'   
        # Save vis images this temp folder, based on this experiment's folder.
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        name = '{:02d}_cat.png'.format(kk)
        cv2.imwrite(save_path + name, cat_img)


def save_scripts(path, scripts_to_save=None):
    if not os.path.exists(os.path.join(path, 'scripts')):
        os.makedirs(os.path.join(path, 'scripts'))

    if scripts_to_save is not None:
        for script in scripts_to_save:
            dst_path = os.path.join(path, 'scripts', script)
            try:
                shutil.copy(script, dst_path)
            except IOError:
                os.makedirs(os.path.dirname(dst_path))
                shutil.copy(script, dst_path)


def torch_tile(a, dim, n_tile):
    """
    This function is taken form PyTorch forum and mimics the behavior of tf.tile.
    Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
    """
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(a.device)

    return torch.index_select(a, dim, order_index)


def reparametrize(mu, logvar):
    std = logvar.mul(0.5).exp_()
    eps = torch.cuda.FloatTensor(std.size()).normal_()
    eps = Variable(eps)

    return eps.mul(std).add_(mu)


def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg


def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)

    return annealed


def make_dis_label(label, gts):
    D_label = torch.ones(gts.shape, device=gts.device, requires_grad=False).float() * label
    return D_label


def sample_p_0(images, opt):
    b, c, h, w = images.shape
    return opt.e_init_sig * torch.randn(*[b, opt.latent_dim]).to(images.device)


def compute_energy(option, score):
    if option.e_energy_form == 'tanh':
        energy = F.tanh(score.squeeze())
    elif option.e_energy_form == 'sigmoid':
        energy = F.sigmoid(score.squeeze())
    elif option.e_energy_form == 'softplus':
        energy = F.softplus(score.squeeze())
    else:
        energy = score.squeeze()
    return energy


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self
    def allowDotting(self, state=True):
        if state:
            self.__dict__ = self
        else:
            self.__dict__ = dict()