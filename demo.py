import os
import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from config import param as option
from model.get_model import get_model
import torchvision.transforms as transforms


def vis_feat(h, w, features):
    vis_feat_list = []
    for i, feature in enumerate(features):
        feat_mean = feature[0].squeeze().mean(0)
        feat = ((feat_mean-feat_mean.min())/(feat_mean.max()-feat_mean.min())).squeeze().detach().cpu().numpy()*255
        feat = cv2.resize(feat, (h//4, w//4), interpolation=cv2.INTER_NEAREST)
        im_color = cv2.applyColorMap(np.array(feat, np.uint8), cv2.COLORMAP_JET)
        vis_feat_list.append(im_color)
    return vis_feat_list


def rgb_loader(path):
    img_transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    with open(path, 'rb') as f:
        img = Image.open(f)
        w, h = img.size
        return img_transform(img.convert('RGB')), (w, h)


def forward_a_sample(model, image, HH, WW, depth=None):
    with torch.no_grad():
        model_pred = model(img=image, depth=depth)
    res, backbone_features = model_pred['sal_pre'][-1], model_pred['backbone_features']
    # Inference and get the last one of the output list
    res = F.upsample(res, size=[WW, HH], mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = 255*(res - res.min()) / (res.max() - res.min() + 1e-8)
    vis_feat_list = vis_feat(HH, WW, backbone_features)
    return res, vis_feat_list

img_path = 'assert/ILSVRC2013_test_00001097.png'
save_path = 'assert/demo_output'
os.makedirs(save_path, exist_ok=True)
img, (h, w) = rgb_loader(img_path)
img = img.unsqueeze(0).cuda()
model, _ = get_model(option)
model.eval()
res, vis_feat_list = forward_a_sample(model, img, h, w)
cv2.imwrite(os.path.join(save_path, 'pred_' + img_path.split('/')[-1]), res)
for i, feature in enumerate(vis_feat_list):
    cv2.imwrite(os.path.join(save_path, str(i) + '_' + img_path.split('/')[-1]), feature)
