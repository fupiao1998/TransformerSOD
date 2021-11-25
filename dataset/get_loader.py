import torch.utils.data as data
from dataset.dataloader import SalObjDatasetRGBD, SalObjDatasetWeak, SalObjDatasetRGB


def get_loader(option, pin_memory=True):
    if option['task'] == 'RGBD-SOD':
        dataset = SalObjDatasetRGBD(option['paths']['image_root'], option['paths']['gt_root'], 
                                    option['paths']['depth_root'], trainsize=option['trainsize'])
    elif option['task'] == 'Weak-RGB-SOD':
        dataset = SalObjDatasetWeak(option['paths']['image_root'], option['paths']['gt_root'], 
                                    option['paths']['mask_root'], option['paths']['gray_root'], 
                                    trainsize=option['trainsize'])
    else:
        dataset = SalObjDatasetRGB(option['paths']['image_root'], option['paths']['gt_root'], trainsize=option['trainsize'])
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=option['batch_size'],
                                  shuffle=True,
                                  num_workers=option['batch_size'],
                                  pin_memory=pin_memory)
    return data_loader, dataset.size
