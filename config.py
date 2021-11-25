import os
import time
import argparse


parser = argparse.ArgumentParser(description='Decide Which Task to Training')
parser.add_argument('--task', type=str, default='SOD', choices=['COD', 'SOD', 'RGBD-SOD', 'Weak-RGB-SOD'])
parser.add_argument('--backbone', type=str, default='swin', choices=['swin', 'R50', 'dpt'])
parser.add_argument('--decoder', type=str, default='simple', choices=['trans', 'rcab', 'simple'])
parser.add_argument('--fusion', type=str, default='early', choices=['early', 'late'])
parser.add_argument('--fusion_method', type=str, default='refine', choices=['refine', 'attention'])
parser.add_argument('--training_path', type=str, default='/home1/maoyuxin/datasets/SOD/DUTS/')
parser.add_argument('--log_info', type=str, default='REMOVE')
parser.add_argument('--neck_channel', type=int, default=64)
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--confiednce_learning', action='store_true')
parser.add_argument('--use_22k', action='store_true')
args = parser.parse_args()

## Configs
param = {}
param['task'] = args.task

## Training Config
param['epoch'] = 50           # 训练轮数
param['seed'] = 1234          # 随机种子 
param['batch_size'] = 4 if param['task']=='Weak-RGB-SOD' else 8       # 批大小
param['save_epoch'] = 5       # 每隔多少轮保存一次模型
param['lr_config'] = {'beta': [0.5, 0.999], 'lr': 2.5e-5, 'lr_dis': 1e-5, 
                      'decay_rate': 0.5, 'decay_epoch': 20, 'gamma': 0.98}
param['trainsize'] = 384      # 训练图片尺寸
param['optim'] = "AdamW"
param['size_rates'] = [1]     # 多尺度训练  [0.75, 1, 1.25]/[1]
## Model Config
# RGB Model
param['neck'] = 'basic'
param['deep_sup'] = False
param['neck_channel'] = args.neck_channel
param['backbone'] = args.backbone
param['decoder'] = args.decoder
# Depth Model
param['fusion'] = args.fusion   # [early, late, cross]
param['fusion_method'] = args.fusion_method

##### uncertainty configs [work in process] #####
param['latent_dim'] = 32   # For vae, it is 8, for abp and gan, it is 32
# param['langevin_step_num_gen'] = 5
# param['sigma_gen'] = 0.3
# param['langevin_s'] = 0.1
##### uncertainty configs [work in process] #####

if args.use_22k:
    param['pretrain'] = "model/swin_base_patch4_window12_384_22k.pth"
else:
    param['pretrain'] = "model/swin_base_patch4_window12_384.pth"
param['confiednce_learning'] = args.confiednce_learning

# Model Config
param['use_attention'] = False
param['scale_trans_radio'] = 0  # Default 0.5
param['rot_trans_radio'] = 0    # Default 0.5

# Backbone Config
param['model_name'] = '{}_{}_{}'.format(param['backbone'], param['neck'], param['decoder'])


# Dataset Config
if param['task'] == 'COD':
    param['image_root'] = '/home1/datasets/SOD_COD/COD/camouflage/COD_train/Imgs/'
    param['gt_root'] = '/home1/datasets/SOD_COD/COD/camouflage/COD_train/GT/'
    param['test_dataset_root'] = '/home1/datasets/SOD_COD/COD/COD_test/'
elif param['task'] == 'SOD':
    param['image_root'] = args.training_path + '/img/'
    param['gt_root'] = args.training_path + '/gt/'
    param['test_dataset_root'] = '/home2/dataset/maoyuxin/SOD_COD/SOD_RGB/'
elif param['task'] == 'RGBD-SOD':
    param['image_root'] = '/home1/maoyuxin/datasets/SOD/RGBD_SOD/train/RGB/'
    param['gt_root'] = '/home1/maoyuxin/datasets/SOD/RGBD_SOD/train/GT/'
    param['depth_root'] = '/home1/maoyuxin/datasets/SOD/RGBD_SOD/train/depth/'
    param['test_dataset_root'] = '/home1/maoyuxin/datasets/SOD/RGBD_SOD/test/'
elif param['task'] == 'Weak-RGB-SOD':
    param['image_root'] = '/home1/datasets/SOD_COD/Scribble_SOD/img/'
    param['gt_root'] = '/home1/datasets/SOD_COD/Scribble_SOD/gt/'
    param['mask_root'] = '/home1/datasets/SOD_COD/Scribble_SOD/mask/'
    param['gray_root'] = '/home1/datasets/SOD_COD/Scribble_SOD/gray/'
    param['test_dataset_root'] = '/home1/datasets/SOD_COD/SOD_RGB/'


# Experiment Dir Config
log_info = param['model_name'] + '_' + args.log_info    # 这个参数可以定义本次实验的名字
param['training_info'] = param['task'] + '_' + str(param['lr_config']['lr']) + '_' + log_info
param['log_path'] = 'experiments/{}'.format(param['training_info'])   # 日志保存路径
param['ckpt_save_path'] = param['log_path'] + '/models/'              # 权重保存路径
print('[INFO] Experiments saved in: ', param['training_info'])


# Test Config
param['testsize'] = 384
if args.ckpt is not None:
    if args.ckpt.lower() == 'last':
        model_path = os.path.join(param['log_path'], 'models')
        model_list = os.listdir(model_path)
        model_list.sort(key=lambda x:int(x[:2]))
        param['checkpoint'] = os.path.join(model_path, model_list[-1])
    else:
        param['checkpoint'] = args.ckpt
else:
    param['checkpoint'] = None

param['eval_save_path'] = param['log_path'] + '/save_images/'         # eval image保存路径
if param['task'] == 'COD':
    param['datasets'] = ['CAMO' , 'CHAMELEON', 'COD10K', 'NC4K']  #'CAMO' , 'CHAMELEON', 'COD10K', 'NC4K'
elif param['task'] == 'SOD':
    param['datasets'] = ['DUTS', 'ECSSD', 'DUT', 'HKU-IS', 'PASCAL', 'SOD']  # , 'THUR', 'HKU-IS', 'MSRA-B', 'PASCAL', 'ECSSD', 'DUT', 'DUTS'
elif param['task'] == 'Weak-RGB-SOD':
    param['datasets'] = ['DUTS', 'ECSSD', 'DUT', 'HKU-IS', 'PASCAL', 'SOD']
elif param['task'] == 'RGBD-SOD':
    param['datasets'] = ['NJU2K', 'STERE', 'DES', 'NLPR', 'LFSD', 'SIP']
