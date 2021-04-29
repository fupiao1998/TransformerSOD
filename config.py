import os
import time
import argparse


parser = argparse.ArgumentParser(description='Decide Which Task to Training')
parser.add_argument('--task', type=str, default='SOD', choices=['COD', 'SOD', 'FIXSOD', 'FIXCOD'])
parser.add_argument('--model', type=str, default='LateFusion', 
                    choices=['DPT', 'VGG', 'ResNet', 'LateFusion', 'CrossFusion', 'DPTDS', 'swin'])
parser.add_argument('--training_path', type=str, default='/home1/datasets/SOD_COD/DUTS/')
parser.add_argument('--log_info', type=str, default='REMOVE')
parser.add_argument('--ckpt', type=str, default='COD')
args = parser.parse_args()

# Configs
param = {}
param['task'] = args.task

# Training Config
param['epoch'] = 50           # 训练轮数
param['seed'] = 1234          # 随机种子 
param['batch_size'] = 8       # 批大小
param['save_epoch'] = 5       # 每隔多少轮保存一次模型
param['lr'] = 2.5e-5          # 学习率
param['trainsize'] = 384      # 训练图片尺寸
param['decay_rate'] = 0.5
param['decay_epoch'] = 20
param['beta'] = [0.5, 0.999]  # Adam参数
param['size_rates'] = [1]     # 多尺度训练  [0.75, 1, 1.25]/[1]
param['use_pretrain'] = True
param['attention_decoder'] = True


# Backbone Config
param['model_name'] = args.model   # [VGG, ResNet, DPT]
param['backbone_name'] = "vitb_rn50_384"   # vitl16_384


# Dataset Config
if param['task'] == 'COD':
    param['image_root'] = '/data/maoyuxin/datasets/COD_datasets/camouflage/COD_train/Imgs/'
    param['gt_root'] = '/data/maoyuxin/datasets/COD_datasets/camouflage/COD_train/GT/'
    param['test_dataset_root'] = '/data/maoyuxin/datasets/COD_datasets/COD_test/'
elif param['task'] == 'SOD':
    param['image_root'] = args.training_path + '/img/'
    param['gt_root'] = args.training_path + '/gt/'
    param['test_dataset_root'] = '/home1/datasets/SOD_COD/SOD_RGB/'
elif param['task'] == 'FIXSOD':
    param['image_root'] = '/data/maoyuxin/datasets/COD_datasets/FixationSOD/images/train/'
    param['gt_root'] = '/data/maoyuxin/datasets/COD_datasets/FixationSOD/gt/train/'
    param['test_dataset_root'] = '/data/maoyuxin/datasets/COD_datasets/FixationSOD/images/val/'
elif param['task'] == 'FIXCOD':
    param['image_root'] = '/data/maoyuxin/datasets/COD_datasets/FixationCOD/train/img/'
    param['gt_root'] = '/data/maoyuxin/datasets/COD_datasets/FixationCOD/train/fix/'
    param['test_dataset_root'] = '/data/maoyuxin/datasets/COD_datasets/FixationCOD/test/img/'


# Experiment Dir Config
log_info = args.model + '_' + args.log_info    # 这个参数可以定义本次实验的名字
param['training_info'] = param['task'] + '_' + param['backbone_name'] + '_' + str(param['lr']) + '_' + log_info
param['log_path'] = 'experiments/{}'.format(param['training_info'])   # 日志保存路径
param['ckpt_save_path'] = param['log_path'] + '/models/'              # 权重保存路径
print('[INFO] Experiments saved in: ', param['training_info'])


# Test Config
param['testsize'] = 352
param['checkpoint'] = args.ckpt
param['eval_save_path'] = param['log_path'] + '/save_images/'         # eval image保存路径
if param['task'] == 'COD':
    param['datasets'] = ['CAMO' , 'CHAMELEON', 'COD10K', 'NC4K']  #'CAMO' , 'CHAMELEON', 'COD10K', 'NC4K'
elif param['task'] == 'SOD':
    param['datasets'] = ['DUTS', 'ECSSD', 'DUT', 'HKU-IS', 'PASCAL', 'SOD']  # , 'THUR', 'HKU-IS', 'MSRA-B', 'PASCAL', 'ECSSD', 'DUT', 'DUTS'
    # param['datasets'] = ['MSRA-B', 'PASCAL', 'SOD']
elif param['task'] == 'FIXSOD':
    param['datasets'] = ['FIXSOD']
elif param['task'] == 'FIXCOD':
    param['datasets'] = ['FIXCOD_Baseline_30']
