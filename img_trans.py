import torch
import random
import numpy as np
import torch.nn.functional as F


def scale_trans(image, sal_list, scale_rate=0.5):
    scale_shape = image.shape[-1]*scale_rate
    scale_shape_corr = int(round(scale_shape / 32) * 32)
    images_scale = F.upsample(image, size=(scale_shape_corr, scale_shape_corr), mode='bilinear', align_corners=True)
    images_scale = F.upsample(images_scale, size=(image.shape[-1], image.shape[-1]), mode='bilinear', align_corners=True)
    sal_list_out = []
    for sal_img in sal_list:
        sal_img_out = F.upsample(sal_img, size=(scale_shape_corr, scale_shape_corr), mode='bilinear', align_corners=True)
        sal_img_out = F.upsample(sal_img_out, size=(image.shape[-1], image.shape[-1]), mode='bilinear', align_corners=True)
        sal_list_out.append(sal_img_out)
    return images_scale, sal_list_out


def rot_trans(image, sal_list):
    def get_rot_mat(theta):
        theta = torch.tensor(theta)
        return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                            [torch.sin(theta), torch.cos(theta), 0]]).cuda()

    def rot_img(x, theta):
        input_type = x.dtype
        rot_mat = get_rot_mat(theta)[None, ...].repeat(x.shape[0], 1, 1).type(input_type)
        grid = F.affine_grid(rot_mat, x.size()).type(input_type).cuda()
        x_out = F.grid_sample(x, grid)
        return x_out

    theta = random.choice((np.pi/2, -np.pi/2, np.pi))
    images_rot = rot_img(image, theta)
    sal_list_out = []
    for sal_img in sal_list:
        sal_img_out = rot_img(sal_img, theta)
        sal_list_out.append(sal_img_out)

    return images_rot, sal_list_out
