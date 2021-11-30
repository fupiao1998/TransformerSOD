import os
import time
import argparse
from path_config import get_path_dict
import socket
hostname = socket.getfqdn(socket.gethostname(  ))

parser = argparse.ArgumentParser(description='Decide Which Task to Training')
parser.add_argument('--task', type=str, default='SOD', choices=['COD', 'SOD', 'RGBD-SOD', 'Weak-RGB-SOD'])
parser.add_argument('--backbone', type=str, default='swin', choices=['swin', 'R50', 'dpt'])
parser.add_argument('--decoder', type=str, default='simple', choices=['trans', 'rcab', 'simple', 'cat'])
parser.add_argument('--fusion', type=str, default='early', choices=['early', 'late'])
parser.add_argument('--loss', type=str, default='structure', choices=['structure', 'bce'])
parser.add_argument('--fusion_method', type=str, default='refine', choices=['refine', 'attention'])
parser.add_argument('--uncer_method', type=str, default='gan', choices=['gan', 'vae', 'abp', 'ebm', 'basic'])
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
param['epoch'] = 50
param['seed'] = 1234
param['batch_size'] = 4 if param['task']=='Weak-RGB-SOD' else 8       # 批大小
param['save_epoch'] = 5
param['lr_config'] = {'beta': [0.5, 0.999], 'lr': 2.5e-5, 'lr_dis': 1e-5, 
                      'decay_rate': 0.5, 'decay_epoch': 20, 'gamma': 0.98}
param['trainsize'] = 384
param['optim'] = "AdamW"
param['loss'] = args.loss
param['size_rates'] = [1] 

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
param['uncer_method'] = args.uncer_method   # gan, vae, abp, ebm
param['vae_config'] = {'reg_weight': 1e-4, 'lat_weight': 1, 'vae_loss_weight': 0.4, 'latent_dim': 8}
param['gan_config'] = {'pred_label': 0, 'gt_label': 1, 'latent_dim': 32}
param['abp_config'] = {'step_num': 5, 'sigma_gen': 0.3, 'langevin_s': 0.1, 'latent_dim': 32}
param['ebm_config'] = {'ebm_out_dim': 1, 'ebm_middle_dim': 100, 'latent_dim': 32, 'e_init_sig': 1.0, 
                       'e_l_steps': 5, 'e_l_step_size': 0.4, 'e_prior_sig': 1.0, 'g_l_steps': 5,
                       'g_llhd_sigma': 0.3, 'g_l_step_size': 0.1, 'e_energy_form': 'identity'}
param['basic_config'] = {'latent_dim': 32}   # Just for placeholder!!!
param['latent_dim'] = param['{}_config'.format(param['uncer_method'])]['latent_dim']
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
param['model_name'] = '{}_{}_{}_{}'.format(param['backbone'], param['neck'], 
                                           param['decoder'], param['uncer_method'])


# Experiment Dir Config
log_info = param['model_name'] + '_' + args.log_info    # 这个参数可以定义本次实验的名字
param['training_info'] = param['task'] + '_' + str(param['lr_config']['lr']) + '_' + log_info
param['log_path'] = 'experiments/{}'.format(param['training_info'])   # 日志保存路径
param['ckpt_save_path'] = param['log_path'] + '/models/'              # 权重保存路径
print('[INFO] Experiments saved in: ', param['training_info'])


# Dataset Config
param['paths'] = get_path_dict(hostname=hostname, task=param['task'])


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
