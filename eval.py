import os
import time
import torch
import argparse
import numpy as np
import pandas as pd
import os.path as osp
import torch.nn as nn
from data import eval_Dataset
# from config import param as option
from torchvision import transforms


def Eval_mae(loader, cuda=True):
        #print('eval[MAE]:{} dataset with {} method.'.format(self.dataset, self.method))
        avg_mae, img_num, total = 0.0, 0.0, 0.0
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            for pred, gt in loader:
                if cuda:
                    pred = trans(pred).cuda()
                    gt = trans(gt).cuda()
                else:
                    pred = trans(pred)
                    gt = trans(gt)
                mea = torch.abs(pred - gt).mean()
                #print(mea)
                #total = total+mea
                if mea == mea: # for Nan
                    avg_mae += mea
                    img_num += 1.0
            avg_mae /= img_num
            #avg = total / img_num
        return avg_mae


def Eval_fmeasure(loader, cuda=True):
    # print('eval[FMeasure]:{} dataset with {} method.'.format(self.dataset, self.method))
        beta2 = 0.3
        avg_f, img_num = 0.0, 0.0

        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            for pred, gt in loader:
                if cuda:
                    pred = trans(pred).cuda()
                    gt = trans(gt).cuda()
                else:
                    pred = trans(pred)
                    gt = trans(gt)
                prec, recall = eval_pr(pred, gt, 255, cuda)
                f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
                f_score[f_score != f_score] = 0  # for Nan
                avg_f += f_score
                img_num += 1.0
            score = avg_f / img_num
        return score.mean()


def Eval_Emeasure(loader,cuda=True):
        # print('eval[EMeasure]:{} dataset with {} method.'.format(self.dataset, self.method))
        avg_e, img_num = 0.0, 0.0
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            scores = torch.zeros(255)
            if cuda:
                scores = scores.cuda()
            for pred, gt in loader:
                if cuda:
                    pred = trans(pred).cuda()
                    gt = trans(gt).cuda()
                else:
                    pred = trans(pred)
                    gt = trans(gt)
                scores += eval_e(pred, gt, 255)
                img_num += 1.0

            scores /= img_num
        # return scores.max()
        return scores.mean()


def Eval_Smeasure(loader,cuda=True):
        # print('eval[SMeasure]:{} dataset with {} method.'.format(self.dataset, self.method))
        alpha, avg_q, img_num = 0.5, 0.0, 0.0
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            for pred, gt in loader:
                if cuda:
                    pred = trans(pred).cuda()
                    gt = trans(gt).cuda()
                else:
                    pred = trans(pred)
                    gt = trans(gt)
                y = gt.mean()
                if y == 0:
                    x = pred.mean()
                    Q = 1.0 - x
                elif y == 1:
                    x = pred.mean()
                    Q = x
                else:
                    gt[gt >= 0.5] = 1
                    gt[gt < 0.5] = 0
                    # print(self._S_object(pred, gt), self._S_region(pred, gt))
                    Q = alpha * S_object(pred, gt) + (1 - alpha) * S_region(pred, gt)
                    if Q.item() < 0:
                        Q = torch.FloatTensor([0.0])
                img_num += 1.0
                avg_q += Q.item()
            avg_q /= img_num
        return avg_q


def eval_e(y_pred, y, num, cuda=True):
    if cuda:
        score = torch.zeros(num).cuda()
        thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
    else:
        score = torch.zeros(num)
        thlist = torch.linspace(0, 1 - 1e-10, num)
    for i in range(num):
        y_pred_th = (y_pred >= thlist[i]).float()
        fm = y_pred_th - y_pred_th.mean()
        gt = y - y.mean()
        align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
        enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4
        score[i] = torch.sum(enhanced) / (y.numel() - 1 + 1e-20)
    return score


def eval_pr(y_pred, y, num, cuda=True):
    
    if cuda:
        prec, recall = torch.zeros(num).cuda(), torch.zeros(num).cuda()
        thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
    else:
        prec, recall = torch.zeros(num), torch.zeros(num)
        thlist = torch.linspace(0, 1 - 1e-10, num)
    for i in range(num):
        y_temp = (y_pred >= thlist[i]).float()
        tp = (y_temp * y).sum()
        prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)
    return prec, recall


def S_object(pred, gt):
    fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
    bg = torch.where(gt == 1, torch.zeros_like(pred), 1 - pred)
    o_fg = object(fg, gt)
    o_bg = object(bg, 1 - gt)
    u = gt.mean()
    Q = u * o_fg + (1 - u) * o_bg
    return Q


def object( pred, gt):
    temp = pred[gt == 1]
    x = temp.mean()
    sigma_x = temp.std()
    score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)

    return score


def S_region( pred, gt):
    X, Y = centroid(gt)
    gt1, gt2, gt3, gt4, w1, w2, w3, w4 = divideGT(gt, X, Y)
    p1, p2, p3, p4 =dividePrediction(pred, X, Y)
    Q1 = ssim(p1, gt1)
    Q2 = ssim(p2, gt2)
    Q3 = ssim(p3, gt3)
    Q4 = ssim(p4, gt4)
    Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
    # print(Q)
    return Q


def centroid( gt ,cuda=True):
    rows, cols = gt.size()[-2:]
    gt = gt.view(rows, cols)
    if gt.sum() == 0:
        if cuda:
            X = torch.eye(1).cuda() * round(cols / 2)
            Y = torch.eye(1).cuda() * round(rows / 2)
        else:
            X = torch.eye(1) * round(cols / 2)
            Y = torch.eye(1) * round(rows / 2)
    else:
        total = gt.sum()
        if cuda:
            i = torch.from_numpy(np.arange(0, cols)).cuda().float()
            j = torch.from_numpy(np.arange(0, rows)).cuda().float()
        else:
            i = torch.from_numpy(np.arange(0, cols)).float()
            j = torch.from_numpy(np.arange(0, rows)).float()
        X = torch.round((gt.sum(dim=0) * i).sum() / total)
        Y = torch.round((gt.sum(dim=1) * j).sum() / total)
    return X.long(), Y.long()


def divideGT(gt, X, Y):
    h, w = gt.size()[-2:]
    area = h * w
    gt = gt.view(h, w)
    LT = gt[:Y, :X]
    RT = gt[:Y, X:w]
    LB = gt[Y:h, :X]
    RB = gt[Y:h, X:w]
    X = X.float()
    Y = Y.float()
    w1 = X * Y / area
    w2 = (w - X) * Y / area
    w3 = X * (h - Y) / area
    w4 = 1 - w1 - w2 - w3
    return LT, RT, LB, RB, w1, w2, w3, w4


def dividePrediction( pred, X, Y):
    h, w = pred.size()[-2:]
    pred = pred.view(h, w)
    LT = pred[:Y, :X]
    RT = pred[:Y, X:w]
    LB = pred[Y:h, :X]
    RB = pred[Y:h, X:w]
    return LT, RT, LB, RB


def ssim( pred, gt):
    gt = gt.float()
    h, w = pred.size()[-2:]
    N = h * w
    x = pred.mean()
    y = gt.mean()
    sigma_x2 = ((pred - x) * (pred - x)).sum() / (N - 1 + 1e-20)
    sigma_y2 = ((gt - y) * (gt - y)).sum() / (N - 1 + 1e-20)
    sigma_xy = ((pred - x) * (gt - y)).sum() / (N - 1 + 1e-20)

    aplha = 4 * x * y * sigma_xy
    beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

    if aplha != 0:
        Q = aplha / (beta + 1e-20)
    elif aplha == 0 and beta == 0:
        Q = 1.0
    else:
        Q = 0
    return Q


task = "SOD"
if task == "SOD":
    gt_dir = "/home1/datasets/SOD_COD/SOD_RGB/"
    test_datasets = ['DUTS', 'ECSSD', 'DUT', 'HKU-IS', 'PASCAL', 'SOD'] # ['DUTS', 'ECSSD', 'DUT', 'HKU-IS', 'THUR', 'SOC']
elif task == "COD":
    gt_dir = "/data/maoyuxin/datasets/COD_datasets/COD_test/"
    test_datasets = ['CAMO', 'CHAMELEON', 'COD10K', 'NC4K']
elif task == "RGBD-SOD":
    gt_dir = "/data/maoyuxin/codes/COD/rgbd_sod/dataset/test/"
    test_datasets = ['NJU2K', 'DES', 'NLPR', 'LFSD', 'SIP', 'STERE']
pred_dir = 'experiments/SOD_vitb_rn50_384_2.5e-05_DPTDS_EPOCH50/save_images/50_epoch/'
print(pred_dir)

latex_str = ""
results_list = []
columns_pd = ['S_measure', 'F_measure', 'E_measure', 'MAE']
for dataset in test_datasets:
    print("[INFO]: Process {} dataset".format(dataset))
    loader = eval_Dataset(osp.join(pred_dir, dataset), osp.join(gt_dir, 'GT', dataset))
    S_measure = Eval_Smeasure(loader=loader, cuda=True)
    F_measure = Eval_fmeasure(loader=loader, cuda=True)
    E_measure = Eval_Emeasure(loader=loader, cuda=True)
    MAE = Eval_mae(loader=loader, cuda=True)
    measure_list = np.array([S_measure, F_measure.item(), E_measure.item(), MAE.item()])
    print(pd.DataFrame(data=np.reshape(measure_list, [1, len(measure_list)]), 
                       columns=columns_pd).to_string(index=False, float_format="%.5f"))
    results_list.append(measure_list)
    latex_str_tmp = '&{:.3f} &{:.3f} &{:.3f} &{:.3f} '.format(S_measure, F_measure, E_measure, MAE)
    latex_str += latex_str_tmp
    print(latex_str_tmp)

result_table = pd.DataFrame(data=np.vstack((results_list)), columns=columns_pd, index=test_datasets)
# import pdb; pdb.set_trace()
with open(pred_dir+'eval_results.csv', 'w') as f:
    result_table.to_csv(f, float_format="%.5f")
print(result_table.to_string(float_format="%.5f"))
print(latex_str)
