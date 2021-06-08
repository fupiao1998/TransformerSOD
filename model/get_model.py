import torch
from model.DPT.DPT import DPTSegmentationModel
from model.DPTDS.DPTDS import DPTSegmentationModelDS
from model.VGG.vgg_models import VGG_Baseline
from model.ResNet.ResNet_models import ResNet_Baseline
from model.Fusion.LateFusionNet import LateFusionSegmentationModel
from model.Fusion.CrossFusionNet import CrossFusionSegmentationModel
from model.swin.swin import Swin
from model.swin.swin_rcab import Swin_rcab
from model.swin.swin_rcab_cross import Swin_rcab_cross
from model.swin.swin import FCDiscriminator


def get_model(option):
    if option['confiednce_learning']:
        dis_model = FCDiscriminator(ndf=64).cuda()
        print("[INFO]: Discriminator have {:.4f}Mb paramerters in total".format(sum(x.numel()/1e6 for x in dis_model.parameters())))
    else:
        dis_model = None
        print("[INFO]: No Discriminator, Only training for Generator!")
    model_name = option['model_name']
    if model_name == 'DPT':
        model = DPTSegmentationModel(1, backbone=option['backbone_name'], use_pretrain=True, use_attention=option['attention_decoder']).cuda()
    elif model_name == 'DPTDS':
        model = DPTSegmentationModelDS(1, backbone=option['backbone_name'], use_pretrain=True).cuda()
    elif model_name == 'ResNet':
        model = ResNet_Baseline(use_pretrain=True).cuda()
    elif model_name == 'VGG':
        model = VGG_Baseline().cuda()
    elif model_name == 'LateFusion':
        model = LateFusionSegmentationModel(1, backbone=option['backbone_name'], use_pretrain=True).cuda()
    elif model_name == 'CrossFusion':
        model = CrossFusionSegmentationModel(1, backbone=option['backbone_name'], use_pretrain=True).cuda()
    elif model_name == 'swin':
        model = Swin(option['trainsize'], use_attention=option['use_attention'], pretrain=option['pretrain']).cuda()
    elif model_name == 'swin_rcab':
        model = Swin_rcab(option['trainsize'], pretrain=option['pretrain']).cuda()
    elif model_name == 'swin_rcab_cross':
        model = Swin_rcab_cross(option['trainsize'], pretrain=option['pretrain']).cuda()
    else:
        print("[ERROR]: No model named {}, please attention!!".format(model_name))
        exit()
    print("[INFO]: Model based on {} have {:.4f}Mb paramerters in total".format(model_name, sum(x.numel()/1e6 for x in model.parameters())))

    if option['checkpoint'] is not None:
        model.load_state_dict(torch.load(option['checkpoint']))
        print('Load checkpoint from {}'.format(option['checkpoint']))

    return model, dis_model
