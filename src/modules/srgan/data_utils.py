import numpy as np
import pandas as pd
from os import listdir
from glob import glob
from os.path import join
import cv2

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.tiff', '.TIFF'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def rec_mk_img_paths(row, prefix):
    return [
        [(prefix + '/{}/Plate{}/{}_s{}_w{}.png').format(row['experiment'], row['plate'], row['well'], site, ch)
         for ch in range(1, 7)]
        for site in [1, 2]
    ]

def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])


def hpa_train_hr_transform(crop_size):
    return Compose([
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])

#
# Abstract image dataset instances
#

# class SRGANDataset(Dataset):
#     def __init__(self, dataset_dir, *args, **kwargs):
#         super(SRGANDataset, self).__init__()
        

#
# HPA-adapted dataset instances
#

class HPATrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(HPATrainDatasetFromFolder, self).__init__()
        self.image_filenames = glob(join(dataset_dir, '*_red.png'))
        self.image_filenames = [ 
            p.replace('_red.png', '_{}.png') 
            for p in self.image_filenames 
        ]

        self.crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.crop_half = self.crop_size // 2
        self.hr_transform = train_hr_transform(self.crop_size)
        self.lr_transform = train_lr_transform(self.crop_size, upscale_factor)

    def __getitem__(self, index):
        
        image = np.stack([ 
            cv2.imread(self.image_filenames[index].format(colour), -1) 
            for colour in ['blue', 'red', 'green', 'yellow']
        ], axis=-1).mean(-1).astype(np.uint8)

#         mask = cv2.imread(self.image_filenames[index].format('centroids_masks'), -1) 
#         coords = np.array(np.where(mask)).T

#         if len(coords):
#             centroid = coords[np.random.randint(len(coords))]
#         else:
#             centroid = np.array([
#                 np.random.randint(mask.shape[0]),
#                 np.random.randint(mask.shape[1]),
#             ])

#         centroid = np.clip(
#             centroid, self.crop_half, np.array(mask.shape) - self.crop_half
#         ).astype(np.int)
#         image = image[
#             centroid[0] - self.crop_half: centroid[0] + self.crop_half,
#             centroid[1] - self.crop_half: centroid[1] + self.crop_half
#         ]
#         image = np.expand_dims(image, -1)
        hr_image = self.hr_transform(Image.fromarray(image))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class HPAValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(HPAValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = glob(join(dataset_dir, '*_red.png'))
        self.image_filenames = [ 
            p.replace('_red.png', '_{}.png') 
            for p in self.image_filenames 
        ]

    def __getitem__(self, index):
        hr_image = np.stack([ 
            cv2.imread(self.image_filenames[index].format(colour), -1) 
            for colour in ['blue', 'red', 'green', 'yellow']
        ], axis=-1).mean(-1).astype(np.uint8)
        hr_image = Image.fromarray(hr_image)
        w, h = hr_image.size

        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)

#
# Recursion-adapted dataset instances
#
 
class RecursionTrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(RecursionTrainDatasetFromFolder, self).__init__()
        
        reference_table = pd.read_csv(dataset_dir + '.csv')
        self.image_filenames = []
        for (idx, row) in reference_table.iterrows():
            self.image_filenames += rec_mk_img_paths(row, dataset_dir)
            
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)
    
    def __getitem__(self, index):
        hr_image_raw = np.stack([
            cv2.imread(img, -1) for img in self.image_filenames[index]
        ], axis=-1).mean(-1).astype(np.uint8)
        hr_image = self.hr_transform(Image.fromarray(hr_image_raw))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image
    
    def __len__(self):
        return len(self.image_filenames)
        
class RecursionValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(RecursionValDatasetFromFolder, self).__init__()
        
        reference_table = pd.read_csv(dataset_dir + '.csv')
        self.image_filenames = []
        for (idx, row) in reference_table.iterrows():
            self.image_filenames += rec_mk_img_paths(row, dataset_dir)
    
        self.upscale_factor = upscale_factor
        
    def __getitem__(self, index):
        hr_image_raw = np.stack([
            cv2.imread(img, -1) for img in self.image_filenames[index]
        ], axis=-1).mean(-1).astype(np.uint8)
        hr_image = Image.fromarray(hr_image_raw)
        w, h = hr_image.size
        
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)
    
    def __len__(self):
        return len(self.image_filenames)

#
# Original SRGAN dataset instances
#

class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)
    
    
class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)
