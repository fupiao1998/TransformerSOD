import numpy as np
import torch.nn.functional as F


def scale_trans(image, sal1, scale_rate=0.3):
    images_scale = F.interpolate(image, scale_factor=scale_rate, mode='bilinear', align_corners=True)
    sal1_s = F.interpolate(sal1, scale_factor=scale_rate, mode='bilinear', align_corners=True)

    return images_scale, sal1_s


def rot_trans(image, sal_list, theta=np.pi/2):
    def get_rot_mat(theta):
        theta = torch.tensor(theta)
        return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                            [torch.sin(theta), torch.cos(theta), 0]])

    def rot_img(x, theta):
        input_type = x.dtype
        rot_mat = get_rot_mat(theta)[None, ...].repeat(x.shape[0], 1, 1).type(input_type)
        grid = F.affine_grid(rot_mat, x.size()).type(input_type)
        x = F.grid_sample(x, grid)
        return x

    images_rot = rot_img(image, theta)
    sal_list_out = []
    for sal_img in sal_list:
        sal_img_out = rot_img(sal_img, theta)
        sal_list_out.append(sal_img_out, theta)

    return images_rot, sal_list_out
