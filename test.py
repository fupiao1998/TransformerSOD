import cv2
import time
import torch
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd
import numpy as np
import pdb, os, argparse
from dataset.dataloader import test_dataset, eval_Dataset, test_dataset_rgbd
from tqdm import tqdm
# from model.DPT import DPTSegmentationModel
from config import param as option
from model.get_model import get_model


def eval_mae(loader, cuda=True):
    avg_mae, img_num, total = 0.0, 0.0, 0.0
    with torch.no_grad():
        for pred, gt in loader:
            if cuda:
                pred, gt = pred.cuda(), gt.cuda()
            else:
                pred, gt = (pred), (gt)
            mae = torch.abs(pred - gt).mean()
            if mae == mae: # for Nan
                avg_mae += mae
                img_num += 1.0
        avg_mae /= img_num
    return avg_mae


# Begin the testing process
generator, discriminator = get_model(option)
generator.load_state_dict(torch.load(option['checkpoint']))
generator.eval()
test_datasets, pre_root = option['datasets'], option['eval_save_path']

time_list, mae_list = [], []
test_epoch_num = option['checkpoint'].split('/')[-1].split('_')[0]
save_path_base = pre_root + test_epoch_num + '_epoch/'
# Begin to inference and save masks
print('========== Begin to inference and save masks ==========')
for dataset in test_datasets:
    save_path = save_path_base + dataset + '/'
    print('[INFO]: Save_path is', save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_root = ''
    if option['task'] == 'COD':
        image_root = option['test_dataset_root'] + dataset + '/Imgs/'
        test_loader = test_dataset(image_root, option['testsize'])
    elif option['task'] == 'SOD':
        image_root = option['test_dataset_root'] + '/Imgs/' + dataset + '/'
        test_loader = test_dataset(image_root, option['testsize'])
    elif option['task'] == 'Weak-RGB-SOD':
        image_root = option['test_dataset_root'] + '/Imgs/' + dataset + '/'
        test_loader = test_dataset(image_root, option['testsize'])
    elif option['task'] == 'RGBD-SOD':
        image_root = option['test_dataset_root'] + dataset + '/RGB/'
        depth_root = option['test_dataset_root'] + dataset + '/depth/'
        test_loader = test_dataset_rgbd(image_root, depth_root, option['testsize'])

    for i in tqdm(range(test_loader.size), desc=dataset):
        if option['task'] == 'SOD' or option['task'] == 'Weak-RGB-SOD' or option['task'] == 'COD':
            image, HH, WW, name = test_loader.load_data()
            image = image.cuda()
            torch.cuda.synchronize()
            start = time.time()
            res = generator.forward(image)
        elif option['task'] == 'RGBD-SOD':
            image, depth, HH, WW, name = test_loader.load_data()
            image, depth = image.cuda(), depth.cuda()
            torch.cuda.synchronize()
            start = time.time()
            res = generator.forward(image, depth)

        res = res[-1]   # Inference and get the last one of the output list
        res = F.upsample(res, size=[WW, HH], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        torch.cuda.synchronize()
        end = time.time()
        time_list.append(end-start)
        res = 255*(res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path+name, res)
    print('[INFO] Avg. Time used in this sequence: {:.4f}s'.format(np.mean(time_list)))

# Begin to evaluate the saved masks
print('========== Begin to evaluate the saved masks ==========')
for dataset in tqdm(test_datasets):
    if option['task'] == 'RGBD-SOD' or option['task'] == 'COD':
        gt_root = option['test_dataset_root'] + dataset + '/GT'
    else:
        gt_root = option['test_dataset_root'] + '/GT/' + dataset + '/'

    loader = eval_Dataset(os.path.join(save_path_base, dataset), gt_root)
    mae = eval_mae(loader=loader, cuda=True)
    mae_list.append(mae.item())

print('--------------- Results ---------------')
results = np.array(mae_list)
results = np.reshape(results, [1, len(results)])
mae_table = pd.DataFrame(data=results, columns=test_datasets)
with open(save_path_base+'results.csv', 'w') as f:
    mae_table.to_csv(f, index=False, float_format="%.4f")
print(mae_table.to_string(index=False))
print('--------------- Results ---------------')

