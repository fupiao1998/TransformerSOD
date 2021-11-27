import torch
from model.saliency_detector import discriminator, ebm_prior
from model.saliency_detector import sod_model, sod_model_with_vae


def get_model(option):
    if option['uncer_method'].lower() == 'vae':
        model = sod_model_with_vae(option=option).cuda()
        uncertainty_model = None
    elif option['uncer_method'].lower() == 'abp':
        model = sod_model(option=option).cuda()
        uncertainty_model = None
    elif option['uncer_method'].lower() == 'gan':
        model = sod_model(option=option).cuda()
        uncertainty_model = discriminator(ndf=64).cuda()
    elif option['uncer_method'].lower() == 'ebm':
        model = sod_model(option=option).cuda()
        uncertainty_model = ebm_prior(option['ebm_config']['ebm_out_dim'], 
                                      option['ebm_config']['ebm_middle_dim'], 
                                      option['ebm_config']['latent_dim']).cuda()
    param_count = sum(x.numel()/1e6 for x in model.parameters())
    print("[INFO]: Model based on [{}] have {:.4f}Mb paramerters in total".format(option['model_name'], param_count))

    if option['checkpoint'] is not None:
        model.load_state_dict(torch.load(option['checkpoint']))
        print('Load checkpoint from {}'.format(option['checkpoint']))

    return model, uncertainty_model
