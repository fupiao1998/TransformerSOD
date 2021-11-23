import torch.utils.data as data
from dataset.dataloader import SalObjDatasetRGBD, SalObjDatasetWeak, SalObjDatasetRGB


def get_loader(option, pin_memory=True):
    if option['task'] == 'RGBD-SOD':
        dataset = SalObjDatasetRGBD(option['image_root'], option['gt_root'], option['depth_root'], trainsize=option['trainsize'])
    elif option['task'] == 'Weak-RGB-SOD':
        dataset = SalObjDatasetWeak(option['image_root'], option['gt_root'], option['mask_root'], option['gray_root'], trainsize=option['trainsize'])
    else:
        dataset = SalObjDatasetRGB(option['image_root'], option['gt_root'], trainsize=option['trainsize'])
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=option['batch_size'],
                                  shuffle=True,
                                  num_workers=option['batch_size'],
                                  pin_memory=pin_memory)
    return data_loader, dataset.size
