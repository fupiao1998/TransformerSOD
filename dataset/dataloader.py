import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
from PIL import ImageEnhance
from dataset.augment import cv_random_flip_rgb, randomCrop_rgb, randomRotation_rgb
from dataset.augment import cv_random_flip_rgbd, randomCrop_rgbd, randomRotation_rgbd
from dataset.augment import cv_random_flip_weak, randomCrop_weak, randomRotation_weak, colorEnhance, randomGaussian, randomPeper


class SalObjDatasetRGB(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        image, gt = cv_random_flip_rgb(image, gt)
        image, gt = randomCrop_rgb(image, gt)
        image, gt = randomRotation_rgb(image, gt)
        image = colorEnhance(image)
        gt = randomPeper(gt)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        return {'image': image, 'gt': gt, 'index': index}

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


class SalObjDatasetRGBD(data.Dataset):
    def __init__(self, image_root, gt_root, depth_root=None, trainsize=352):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
                       or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose(
            [transforms.Resize((self.trainsize, self.trainsize)), transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        depth = self.binary_loader(self.depths[index])
        image, gt, depth = cv_random_flip_rgbd(image, gt, depth)
        image, gt, depth = randomCrop_rgbd(image, gt, depth)
        image, gt, depth = randomRotation_rgbd(image, gt, depth)
        image = colorEnhance(image)
        gt = randomPeper(gt)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        depth = self.depths_transform(depth)

        return {'image': image, 'gt': gt, 'depth': depth, 'index': index}

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images)
        images = []
        gts = []
        depths = []
        for img_path, gt_path, depth_path in zip(self.images, self.gts, self.depths):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            depth = Image.open(depth_path)
            if img.size == gt.size and gt.size == depth.size:
                images.append(img_path)
                gts.append(gt_path)
                depths.append(depth_path)
        self.images = images
        self.gts = gts
        self.depths = depths

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt, depth):
        assert img.size == gt.size and gt.size == depth.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), depth.resize((w, h), Image.NEAREST)
        else:
            return img, gt, depth

    def __len__(self):
        return self.size


class SalObjDatasetWeak(data.Dataset):
    def __init__(self, image_root, gt_root, mask_root, gray_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.masks = [mask_root + f for f in os.listdir(mask_root) if f.endswith('.png')]
        self.grays = [gray_root + f for f in os.listdir(gray_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.masks = sorted(self.masks)
        self.grays = sorted(self.grays)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.gray_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        mask = self.binary_loader(self.masks[index])
        gray = self.binary_loader(self.grays[index])
        image, gt, mask, gray = cv_random_flip_weak(image, gt, mask, gray)
        image, gt, mask, gray = randomCrop_weak(image, gt, mask, gray)
        image, gt, mask, gray = randomRotation_weak(image, gt, mask, gray)
        image = colorEnhance(image)
        # gt=randomGaussian(gt)
        gt = randomPeper(gt)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        mask = self.mask_transform(mask)
        gray = self.gray_transform(gray)

        return {'image': image, 'gt': gt, 'mask': mask, 'gray': gray, 'index': index}

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images, gts, masks, grays = [], [], [], []
        for img_path, gt_path, mask_path, gray_path in zip(self.images, self.gts, self.masks, self.grays):
            img, gt, mask, gray = Image.open(img_path), Image.open(gt_path), Image.open(mask_path), Image.open(gray_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
                masks.append(mask_path)
                grays.append(gray_path)
        self.images = images
        self.gts = gts
        self.masks = masks
        self.grays = grays

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def depth_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('I')

    def resize(self, img, gt, mask, gray):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), mask.resize((w, h), Image.NEAREST), gray.resize((w, h), Image.NEAREST)
        else:
            return img, gt, mask, gray

    def __len__(self):
        return self.size


class test_dataset:
    def __init__(self, image_root, testsize):
        self.testsize = testsize
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        HH = image.size[0]
        WW = image.size[1]
        image = self.transform(image).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, None, HH, WW, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


class test_dataset_rgbd:
    def __init__(self, image_root, testsize):
        depth_root = image_root[:-3] + 'GT'
        self.testsize = testsize
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.depths = [os.path.join(depth_root, f) for f in os.listdir(depth_root) if f.endswith('.bmp')
                       or f.endswith('.png')]
        self.images = sorted(self.images)
        self.depths = sorted(self.depths)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # self.gt_transform = transforms.Compose([
        #     transforms.Resize((self.trainsize, self.trainsize)),
        #     transforms.ToTensor()])
        self.depths_transform = transforms.Compose(
            [transforms.Resize((self.testsize, self.testsize)), transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        HH = image.size[0]
        WW = image.size[1]
        image = self.transform(image).unsqueeze(0)
        depth = self.binary_loader(self.depths[self.index])
        depth = self.depths_transform(depth).unsqueeze(0)

        name = self.images[self.index].split('/')[-1]
        # image_for_post=self.rgb_loader(self.images[self.index])
        # image_for_post=image_for_post.resize(gt.size)
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image, depth, HH, WW, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


# class eval_Dataset(data.Dataset):
#     def __init__(self, img_root, label_root):
#         lst_label = sorted(os.listdir(label_root))
#         # print(label_root)
#         lst_pred = sorted(os.listdir(img_root))
#         # print(img_root)
#         lst = []
#         for name in lst_label:
#             if name in lst_pred:
#                 lst.append(name)

#         self.image_path = list(map(lambda x: os.path.join(img_root, x), lst))
#         self.label_path = list(map(lambda x: os.path.join(label_root, x), lst))

#     def __getitem__(self, item):
#         pred = Image.open(self.image_path[item]).convert('L')
#         gt = Image.open(self.label_path[item]).convert('L')
#         if pred.size != gt.size:
#             pred = pred.resize(gt.size, Image.BILINEAR)
#         return pred, gt

#     def __len__(self):
#         return len(self.image_path)

        

class eval_Dataset(data.Dataset):
    def __init__(self, img_root, label_root):
        lst_label = sorted(os.listdir(label_root))
        # print(label_root)
        lst_pred = sorted(os.listdir(img_root))
        # print(img_root)
        self.label_abbr, self.pred_abbr = lst_label[0].split('.')[-1], lst_pred[0].split('.')[-1]
        label_list, pred_list = [], []
        for name in lst_label:
            label_name = name.split('.')[0]
            if label_name+'.'+self.label_abbr in lst_label:
                label_list.append(name)
    
        for name in lst_pred:
            label_name = name.split('.')[0]
            if label_name+'.'+self.pred_abbr in lst_pred:
                pred_list.append(name)

        self.image_path = list(map(lambda x: os.path.join(img_root, x), pred_list))
        self.label_path = list(map(lambda x: os.path.join(label_root, x), label_list))
        self.trans = transforms.Compose([transforms.ToTensor()])

    def get_img_pil(self, path):
        img = Image.open(path).convert('L')
        return img

    def __getitem__(self, item):
        img_path = self.image_path[item]
        label_path = self.label_path[item]
        pred = self.get_img_pil(img_path)  # (500, 375)
        gt = self.get_img_pil(label_path)
        if pred.size != gt.size:
            pred = pred.resize(gt.size, Image.BILINEAR)

        
        return self.trans(pred), self.trans(gt)

    def __len__(self):
        return len(self.image_path)