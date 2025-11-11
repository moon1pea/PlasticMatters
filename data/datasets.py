# # -*- coding: utf-8 -*-

# import os
# from torchvision import transforms
# from torch.utils.data import Dataset
# from PIL import Image, ImageFile
# import random
# import torch
# import kornia.augmentation as K
# from .dct import DCT_base_Rec_Module

# ImageFile.LOAD_TRUNCATED_IMAGES = True

# # Kornia perturbations
# Perturbations = K.container.ImageSequential(
#     K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 3.0), p=0.1),
#     K.RandomJPEG(jpeg_quality=(30, 100), p=0.1)
# )

# transform_before = transforms.Compose([
#     transforms.Resize((256, 256)),   # ✅ 先强制 resize
#     transforms.ToTensor(),
#     transforms.Lambda(lambda x: Perturbations(x)[0])
# ])

# transform_before_test = transforms.Compose([transforms.ToTensor()])

# transform_train = transforms.Compose([
#     transforms.Resize([256, 256]),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])


# def load_data_from_folder(folder_path):
#     """只读取 0_real 和 1_fake 下的图片，如果没有这两个文件夹直接报错"""
#     real_dir = os.path.join(folder_path, '0_real')
#     fake_dir = os.path.join(folder_path, '1_fake')

#     if not os.path.isdir(real_dir) or not os.path.isdir(fake_dir):
#         raise FileNotFoundError(f"'0_real' or '1_fake' folder missing in {folder_path}")

#     data_list = []
#     for img in os.listdir(real_dir):
#         data_list.append({"image_path": os.path.join(real_dir, img), "label": 0})
#     for img in os.listdir(fake_dir):
#         data_list.append({"image_path": os.path.join(fake_dir, img), "label": 1})
#     return data_list


# class TrainDataset(Dataset):
#     def __init__(self, is_train, args):
#         root = args.data_path if is_train else args.eval_data_path
#         self.data_list = load_data_from_folder(root)
#         self.dct = DCT_base_Rec_Module()

#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, index):
#         sample = self.data_list[index]
#         image_path, label = sample['image_path'], sample['label']

#         try:
#             image = Image.open(image_path).convert('RGB')
#         except:
#             print(f'image error: {image_path}')
#             return self.__getitem__(random.randint(0, len(self.data_list) - 1))

#         image = transform_before(image)

#         try:
#             x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct(image)
#         except:
#             print(f'DCT error: {image_path}, shape: {image.shape}')
#             return self.__getitem__(random.randint(0, len(self.data_list) - 1))

#         x_0 = transform_train(image)
#         x_minmin = transform_train(x_minmin)
#         x_maxmax = transform_train(x_maxmax)
#         x_minmin1 = transform_train(x_minmin1)
#         x_maxmax1 = transform_train(x_maxmax1)

#         return torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0], dim=0), torch.tensor(int(label))


# class TestDataset(Dataset):
#     def __init__(self, is_train, args):
#         root = args.data_path if is_train else args.eval_data_path
#         self.data_list = load_data_from_folder(root)
#         self.dct = DCT_base_Rec_Module()

#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, index):
#         sample = self.data_list[index]
#         image_path, label = sample['image_path'], sample['label']

#         image = Image.open(image_path).convert('RGB')
#         image = transform_before_test(image)

#         x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct(image)

#         x_0 = transform_train(image)
#         x_minmin = transform_train(x_minmin)
#         x_maxmax = transform_train(x_maxmax)
#         x_minmin1 = transform_train(x_minmin1)
#         x_maxmax1 = transform_train(x_maxmax1)

#         return torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0], dim=0), torch.tensor(int(label))
# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image
import io
import torch
from .dct import DCT_base_Rec_Module
import random

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import kornia.augmentation as K

Perturbations = K.container.ImageSequential(
    K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 3.0), p=0.1),
    K.RandomJPEG(jpeg_quality=(30, 100), p=0.1)
)

def random_patch_shuffle(img, patch_size=0):
    if not torch.is_tensor(img):
        raise TypeError(f"expected Tensor input but got {type(img)}")
    if img.dim() != 3:
        raise ValueError(f"expected 3D tensor (C, H, W) but got shape {img.shape}")
    c, h, w = img.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(f"image size ({h}, {w}) not divisible by patch_size {patch_size}")
    grid_h = h // patch_size
    grid_w = w // patch_size
    patches = img.view(c, grid_h, patch_size, grid_w, patch_size)
    patches = patches.permute(0, 1, 3, 2, 4).contiguous().view(c, grid_h * grid_w, patch_size, patch_size)
    shuffle_idx = torch.randperm(grid_h * grid_w)
    patches = patches[:, shuffle_idx]
    patches = patches.view(c, grid_h, grid_w, patch_size, patch_size).permute(0, 1, 3, 2, 4).contiguous()
    return patches.view(c, h, w)


transform_before = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: random_patch_shuffle(x, patch_size=16)),
    transforms.Lambda(lambda x: Perturbations(x)[0])
])
transform_before_test = transforms.Compose([
    transforms.ToTensor(),
    ]
)

transform_train = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

transform_test_normalize = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)


class TrainDataset(Dataset):
    def __init__(self, is_train, args):
        
        root = args.data_path if is_train else args.eval_data_path

        self.data_list = []

        if'GenImage' in root and root.split('/')[-1] != 'train':
            file_path = root
            file_path_list = os.listdir(file_path)

            if '0_real' not in file_path_list:
                # Nested structure
                for folder_name in os.listdir(file_path):
                    folder_path = os.path.join(file_path, folder_name)
                    
                    # Skip if it's a file (not a directory)
                    if not os.path.isdir(folder_path):
                        continue
                    
                    folder_contents = os.listdir(folder_path)
                    
                    # Only process if it contains both 0_real and 1_fake
                    if '0_real' in folder_contents and '1_fake' in folder_contents:
                        for image_path in os.listdir(os.path.join(folder_path, '0_real')):
                            if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                                self.data_list.append({"image_path": os.path.join(folder_path, '0_real', image_path), "label" : 0})
                        for image_path in os.listdir(os.path.join(folder_path, '1_fake')):
                            if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                                self.data_list.append({"image_path": os.path.join(folder_path, '1_fake', image_path), "label" : 1})
            
            else:
                # Direct structure: root/0_real/ and root/1_fake/
                for image_path in os.listdir(os.path.join(file_path, '0_real')):
                    if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.data_list.append({"image_path": os.path.join(file_path, '0_real', image_path), "label" : 0})
                for image_path in os.listdir(os.path.join(file_path, '1_fake')):
                    if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.data_list.append({"image_path": os.path.join(file_path, '1_fake', image_path), "label" : 1})
        else:
            # Check if root directly contains 0_real and 1_fake folders
            root_list = os.listdir(root)
            if '0_real' in root_list and '1_fake' in root_list:
                # Structure: root/0_real/ and root/1_fake/
                for image_path in os.listdir(os.path.join(root, '0_real')):
                    if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.data_list.append({"image_path": os.path.join(root, '0_real', image_path), "label" : 0})
                for image_path in os.listdir(os.path.join(root, '1_fake')):
                    if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.data_list.append({"image_path": os.path.join(root, '1_fake', image_path), "label" : 1})
            else:
                # Traverse subdirectories
                for filename in os.listdir(root):
                    file_path = os.path.join(root, filename)
                    
                    # Skip if it's a file (not a directory)
                    if not os.path.isdir(file_path):
                        continue
                    
                    file_path_list = os.listdir(file_path)
                    
                    # Check if this directory directly contains 0_real and 1_fake
                    if '0_real' in file_path_list and '1_fake' in file_path_list:
                        # Structure: root/folder_name/0_real/ and root/folder_name/1_fake/
                        for image_path in os.listdir(os.path.join(file_path, '0_real')):
                            if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                                self.data_list.append({"image_path": os.path.join(file_path, '0_real', image_path), "label" : 0})
                        for image_path in os.listdir(os.path.join(file_path, '1_fake')):
                            if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                                self.data_list.append({"image_path": os.path.join(file_path, '1_fake', image_path), "label" : 1})
                    else:
                        # Check nested structure: root/folder_name/subfolder/0_real/ and 1_fake/
                        for folder_name in file_path_list:
                            folder_path = os.path.join(file_path, folder_name)
                            
                            # Skip if it's a file (not a directory)
                            if not os.path.isdir(folder_path):
                                continue
                            
                            folder_contents = os.listdir(folder_path)
                            
                            # Only process if it contains both 0_real and 1_fake
                            if '0_real' in folder_contents and '1_fake' in folder_contents:
                                for image_path in os.listdir(os.path.join(folder_path, '0_real')):
                                    if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                                        self.data_list.append({"image_path": os.path.join(folder_path, '0_real', image_path), "label" : 0})
                                for image_path in os.listdir(os.path.join(folder_path, '1_fake')):
                                    if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                                        self.data_list.append({"image_path": os.path.join(folder_path, '1_fake', image_path), "label" : 1})
                
        self.dct = DCT_base_Rec_Module()


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        
        sample = self.data_list[index]
                
        image_path, targets = sample['image_path'], sample['label']

        try:
            image = Image.open(image_path).convert('RGB')
        except:
            print(f'image error: {image_path}')
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))


        image = transform_before(image)

        try:
            x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct(image)
        except:
            print(f'image error: {image_path}, c, h, w: {image.shape}')
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))

        x_0 = transform_train(image)
        x_minmin = transform_train(x_minmin) 
        x_maxmax = transform_train(x_maxmax)

        x_minmin1 = transform_train(x_minmin1) 
        x_maxmax1 = transform_train(x_maxmax1)
        


        return torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0], dim=0), torch.tensor(int(targets))

    

class TestDataset(Dataset):
    def __init__(self, is_train, args):
        
        root = args.data_path if is_train else args.eval_data_path

        self.data_list = []

        file_path = root
        file_path_list = os.listdir(file_path)

        if '0_real' not in file_path_list:
            # Nested structure: root/folder_name/0_real/ and 1_fake/
            for folder_name in os.listdir(file_path):
                folder_path = os.path.join(file_path, folder_name)
                
                # Skip if it's a file (not a directory)
                if not os.path.isdir(folder_path):
                    continue
                
                folder_contents = os.listdir(folder_path)
                
                # Only process if it contains both 0_real and 1_fake
                if '0_real' in folder_contents and '1_fake' in folder_contents:
                    for image_path in os.listdir(os.path.join(folder_path, '0_real')):
                        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.data_list.append({"image_path": os.path.join(folder_path, '0_real', image_path), "label" : 0})
                    for image_path in os.listdir(os.path.join(folder_path, '1_fake')):
                        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.data_list.append({"image_path": os.path.join(folder_path, '1_fake', image_path), "label" : 1})
        
        else:
            # Direct structure: root/0_real/ and root/1_fake/
            for image_path in os.listdir(os.path.join(file_path, '0_real')):
                if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.data_list.append({"image_path": os.path.join(file_path, '0_real', image_path), "label" : 0})
            for image_path in os.listdir(os.path.join(file_path, '1_fake')):
                if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.data_list.append({"image_path": os.path.join(file_path, '1_fake', image_path), "label" : 1})


        self.dct = DCT_base_Rec_Module()


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        
        sample = self.data_list[index]
                
        image_path, targets = sample['image_path'], sample['label']

        image = Image.open(image_path).convert('RGB')

        image = transform_before_test(image)

        # x_max, x_min, x_max_min, x_minmin = self.dct(image)

        x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct(image)


        x_0 = transform_train(image)
        x_minmin = transform_train(x_minmin) 
        x_maxmax = transform_train(x_maxmax)

        x_minmin1 = transform_train(x_minmin1) 
        x_maxmax1 = transform_train(x_maxmax1)
        
        return torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0], dim=0), torch.tensor(int(targets))

