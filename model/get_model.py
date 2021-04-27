from model.DPT.DPT import DPTSegmentationModel
from model.DPTDS.DPTDS import DPTSegmentationModelDS
from model.VGG.vgg_models import VGG_Baseline
from model.ResNet.ResNet_models import ResNet_Baseline
from model.Fusion.LateFusionNet import LateFusionSegmentationModel
from model.Fusion.CrossFusionNet import CrossFusionSegmentationModel


def get_model(option):
    model_name = option['model_name']
    if model_name == 'DPT':
        model = DPTSegmentationModel(1, backbone=option['backbone_name'], use_pretrain=option['use_pretrain']).cuda()
    elif model_name == 'DPTDS':
        model = DPTSegmentationModelDS(1, backbone=option['backbone_name'], use_pretrain=option['use_pretrain']).cuda()
    elif model_name == 'ResNet':
        model = ResNet_Baseline(use_pretrain=option['use_pretrain'])
    elif model_name == 'VGG':
        model = VGG_Baseline()
    elif model_name == 'LateFusion':
        model = LateFusionSegmentationModel(1, backbone=option['backbone_name'], use_pretrain=option['use_pretrain']).cuda()
    elif model_name == 'CrossFusion':
        model = CrossFusionSegmentationModel(1, backbone=option['backbone_name'], use_pretrain=option['use_pretrain']).cuda()
    else:
        print("[ERROR]: No model named {}, please attention!!".format(model_name))
        exit()
    print("Model based on {} have {:.4f}Mb paramerters in total".format(model_name, sum(x.numel()/1e6 for x in model.parameters())))

    return model.cuda()
