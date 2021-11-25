import torch
from model.saliency_detector import Discriminator
from model.saliency_detector import sod_model, sod_model_with_vae


def get_model(option):
    if option['confiednce_learning']:
        dis_model = Discriminator(ndf=64).cuda()
        print("[INFO]: Discriminator have {:.4f}Mb paramerters in total".format(sum(x.numel()/1e6 for x in dis_model.parameters())))
    else:
        dis_model = None
        print("[INFO]: No Discriminator, Only training for Generator!")
    if option['uncer_method'].lower() == 'vae':
        model = sod_model_with_vae(option=option).cuda()
    else:
        model = sod_model(option=option).cuda()

    print("[INFO]: Model based on [{}] have {:.4f}Mb paramerters in total".format(option['model_name'], sum(x.numel()/1e6 for x in model.parameters())))

    if option['checkpoint'] is not None:
        model.load_state_dict(torch.load(option['checkpoint']))
        print('Load checkpoint from {}'.format(option['checkpoint']))

    return model, dis_model
