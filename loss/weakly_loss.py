import torch
from loss.lscloss import LocalSaliencyCoherence
from loss.smoothness import smoothness_loss
from loss.StructureConsistency import SaliencyStructureConsistency as ConsistLoss
from img_trans import rot_trans


class weakly_loss():
    def __init__(self):
        self.lsc_loss = LocalSaliencyCoherence()
        self.smoothness_loss = smoothness_loss(size_average=True)
        self.lsc_kernels = [{"weight": 1, "xy": 6, "rgb": 0.1}]
        self.cross_entropy = torch.nn.BCELoss()
        self.lamda = [1, 0.3, 1]
        print('[INFO]: Weakly loss params [{}]'.format(self.lamda))

    def __call__(self, images, outputs, gt, masks, grays, model=None):
        img_size = images.size(2) * images.size(3) * images.size(0)
        ratio = img_size / torch.sum(masks)

        if isinstance(outputs, list):
            loss = 0
            for output in outputs:
                loss_lsc_i = self.lsc_loss(torch.sigmoid(output), self.lsc_kernels, kernels_radius=5, 
                                           sample={'rgb': images.clone()}, height_input=images.shape[2], 
                                           width_input=images.shape[3])['loss']
                loss_smooth_i = self.smoothness_loss(torch.sigmoid(output), grays)
                loss_sal_i = ratio * self.cross_entropy(torch.sigmoid(output)*masks, gt*masks)
                loss_i = self.lamda[0]*loss_lsc_i + self.lamda[1]*loss_smooth_i + self.lamda[2]*loss_sal_i
                loss += loss_i
                
        if model is not None:
            images_rot, outputs_rot = rot_trans(images, outputs)
            outputs_reference = model(img=images_rot.detach())['sal_pre']
            consist_loss = ConsistLoss(torch.sigmoid(outputs_rot[0]), torch.sigmoid(outputs_reference[0]))
            loss = loss + consist_loss

        return loss
