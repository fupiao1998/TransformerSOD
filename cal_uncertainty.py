from tqdm import tqdm
import numpy as np
import cv2
import os


def calculate(img_list):
    mean = np.zeros_like(img_list[0])
    for img in img_list:
        mean += img
    mean = mean / len(img_list)
    predictive = -mean * np.log(np.minimum(mean+1e-8, 1))
    predictive_norm = (predictive-predictive.min()) / (predictive.max()-predictive.min())*255
    predictive_norm_color = cv2.applyColorMap(np.array(predictive_norm, np.uint8), cv2.COLORMAP_JET)
    
    return predictive_norm_color


dataset_list = ['DUT', 'DUTS', 'ECSSD', 'HKU-IS', 'PASCAL', 'SOD']
root = 'experiments/SOD_2.5e-05_swin_basic_cat_ganabp_[lamda_dis_0.2]/save_images'
for dataset in dataset_list:    
    name_list = os.listdir(os.path.join(root, '50_epoch_0', dataset))
    print('[INFO]: Process [{}]'.format(dataset))
    for name in tqdm(name_list):
        img_list = []
        for i in range(10):
            img_root = os.path.join(root, '50_epoch_{}'.format(i), dataset, name)
            img = cv2.imread(img_root).astype(np.float64) / 255.0
            img_list.append(img)
        predictive_norm_color = calculate(img_list)
        save_root = os.path.join(img_root.split('/')[0], img_root.split('/')[1], 'uncertainty', dataset)
        os.makedirs(save_root, exist_ok=True)
        cv2.imwrite(os.path.join(save_root, img_root.split('/')[-1]), predictive_norm_color)
