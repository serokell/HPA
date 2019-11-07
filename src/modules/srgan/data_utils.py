import numpy as np
import pandas as pd
import torch
import cv2
import torchvision.utils as utils

from os import listdir
from glob import glob
from os.path import join
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

def read_img_hpa(img_path_tpl):
    return np.stack([ 
        cv2.imread(img_path_tpl.format(colour), -1) 
        for colour in ['blue', 'red', 'green', 'yellow']
    ], axis=-1)
            
def crop_nonblack(image, mask, crop_half):
    coords = np.array(np.where(mask))[:2].T

    if len(coords):
        centroid = coords[np.random.randint(len(coords))]
    else:
        centroid = np.array([
            np.random.randint(image.shape[0]),
            np.random.randint(image.shape[1]),
        ])

    centroid = np.clip(
        centroid, crop_half, np.array(image.shape)[:2] - crop_half
    ).astype(np.int)
    image = image[
        centroid[0] - crop_half: centroid[0] + crop_half,
        centroid[1] - crop_half: centroid[1] + crop_half
    ]
    if len(image.shape) == 2:
        image = np.expand_dims(image, -1)
    return image


def centr_crop(image, crop_size):
    centroid = np.array([
        image.shape[0] // 2,
        image.shape[1] // 2
    ])

    image = image[
        centroid[0] - crop_size // 2: centroid[0] + crop_size // 2,
        centroid[1] - crop_size // 2: centroid[1] + crop_size // 2
    ]
    if len(image.shape) == 2:
        image = np.expand_dims(image, -1)
    return image

def display_transform(img_tensor):
    img = np.transpose(img_tensor.numpy(), (1,2,0))
    w, h = img.shape[:2]
    if w > h:
        new_w, new_h = int(w * 400 / h), 400
    else:
        new_w, new_h = 400, int(h * 400 / w)
        
    img = cv2.resize(img, (new_w, new_h))
    return ToTensor()(centr_crop(img, 400))

def downsize_img(image, downscale_factor):
    if downscale_factor == 1:
        return image
    return cv2.resize(
        image, (image.shape[0] // downscale_factor, image.shape[1] // downscale_factor),
        interpolation=cv2.INTER_CUBIC)      

def partial_channel_merge(image, merged_channels, not_merged_channels):
    return np.concatenate((image[:, :, not_merged_channels],
                           np.expand_dims(image[:, :, merged_channels].mean(-1), -1)),
                          axis=2)

def normalize_img(image):
    return (image - image.min()) / (image.max() - image.min())

class ImageStacker:
    def __init__(self, cols, rows):
        self.cols = cols
        self.rows = rows
        self.images = []
    
    def add_row(self, *imgs):
        assert len(imgs) == self.cols
        processed_imgs = [
            display_transform(img.data.cpu().squeeze(0))
            for img in imgs
        ]
        self.images.extend(processed_imgs)
    
    def grids(self):
        stacked = torch.stack(self.images)
        chunked = torch.chunk(stacked, stacked.size(0) // (self.cols * self.rows))
        return [utils.make_grid(img[:, :3], nrow=self.cols, padding=5) for img in chunked]
    
#
# HPA-adapted dataset instances
#

_hpa_channel_idxs = {
    'blue': 0,
    'red': 1,
    'green': 2,
    'yellow': 3,
}

class HPATrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor, colorisation=False,
                 merged_channels=None, normalization=True):
        super().__init__()
        self.image_filenames = glob(join(dataset_dir, '*_red.png'))
        self.image_filenames = [ 
            p.replace('_red.png', '_{}.png') 
            for p in self.image_filenames 
        ]
        self.colorisation = colorisation
        self.normalization = normalization
        
        if merged_channels is not None:
            self.merged_channels = [_hpa_channel_idxs[ch] for ch in merged_channels]
            self.not_merged_channels = [ch for ch in range(0, 4) if ch not in self.merged_channels]
        else:
            self.merged_channels = None
           
        self.upscale_factor = upscale_factor
        self.crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.crop_half = self.crop_size // 2

    def __getitem__(self, index):
        image = np.stack([ 
            cv2.imread(self.image_filenames[index].format(colour), -1) 
            for colour in ['blue', 'red', 'green', 'yellow']
        ], axis=-1)
        if not self.colorisation:
            image = image.mean(-1).astype(np.uint8)
        elif self.merged_channels is not None:
            image = partial_channel_merge(image, self.merged_channels, self.not_merged_channels)

        mask = cv2.imread(self.image_filenames[index].format('centroids_masks'), -1) 
        image = crop_nonblack(image, mask, self.crop_half)
        
        hr_image = normalize_img(image) if self.normalization else image
            
        lr_image = downsize_img(hr_image, self.upscale_factor)
        if self.colorisation:
            image = np.expand_dims(lr_image.astype(np.float).sum(-1), -1)
            lr_image = normalize_img(image) if self.normalization else image

        return ToTensor()(lr_image).float(), ToTensor()(hr_image).float()

    def __len__(self):
        return len(self.image_filenames)


class HPAValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor, colorisation=False,
                 merged_channels=None, normalization=True):
        super().__init__()
        self.colorisation = colorisation
        self.normalization = normalization
        
        if merged_channels is not None:
            self.merged_channels = [_hpa_channel_idxs[ch] for ch in merged_channels]
            self.not_merged_channels = [ch for ch in range(0, 4) if ch not in self.merged_channels]
        else:
            self.merged_channels = None
            
        self.upscale_factor = upscale_factor
        self.image_filenames = glob(join(dataset_dir, '*_red.png'))
        self.image_filenames = [ 
            p.replace('_red.png', '_{}.png') 
            for p in self.image_filenames 
        ]

    def __getitem__(self, index):
        image = np .stack([ 
            cv2.imread(self.image_filenames[index].format(colour), -1) 
            for colour in ['blue', 'red', 'green', 'yellow']
        ], axis=-1)
        if not self.colorisation:
            image = image.mean(-1).astype(np.uint8)
        elif self.merged_channels is not None:
            image = partial_channel_merge(image, self.merged_channels, self.not_merged_channels)
    
        #hr_image = Image.fromarray(hr_image)
        hr_image = normalize_img(image) if self.normalization else image
        w, h = hr_image.shape[:2]

        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        #lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        #hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        #hr_image = CenterCrop(crop_size)(hr_image)
        hr_image = centr_crop(hr_image, crop_size)
        lr_image = downsize_img(hr_image, self.upscale_factor)

        if self.colorisation:
            image = np.expand_dims(lr_image.astype(np.float).mean(-1), -1)
            lr_image = normalize_img(image) if self.normalization else image

        if self.upscale_factor != 1:
            hr_restore_img = cv2.resize(lr_image, (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)
        else:
            hr_restore_img = lr_image.squeeze(-1)
         
        if self.colorisation:
            ch_count = hr_image.shape[-1]
            hr_restore_img = np.stack([hr_restore_img]*ch_count, axis=2)
            
        return ToTensor()(lr_image).float(), ToTensor()(hr_restore_img).float(), ToTensor()(hr_image).float()

    def __len__(self):
        return len(self.image_filenames)

#
# Recursion-adapted dataset instances
#

_rx_channel_idxs = {
    'nuclei': 0,
    'endoplasmic_reticuli': 1,
    'actin': 2,
    'nucleoli': 3,
    'mitochondria': 4,
    'golgi': 5
}

class RecursionTrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor, colorisation=False,
                 merged_channels=None, normalization=True):
        super().__init__()
        
        reference_table = pd.read_csv(dataset_dir + '.csv')
        self.image_filenames = []
        for (idx, row) in reference_table.iterrows():
            self.image_filenames += rec_mk_img_paths(row, dataset_dir)
        
        self.colorisation = colorisation
        self.normalization = normalization
        if merged_channels is not None:
            self.merged_channels = [_rx_channel_idxs[ch] for ch in merged_channels]
            self.not_merged_channels = [ch for ch in range(0, 6) if ch not in self.merged_channels]
        else:
            self.merged_channels = None
        
        self.crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.upscale_factor = upscale_factor
    
    def __getitem__(self, index):
        image = np.stack([
            cv2.imread(img, -1) for img in self.image_filenames[index]
        ], axis=-1)
        if not self.colorisation:
            image = image.mean(-1).astype(np.uint8)
        elif self.merged_channels is not None:
            image = partial_channel_merge(image, self.merged_channels, self.not_merged_channels)
        
        mask = cv2.imread(self.image_filenames[index][0].replace('w1', 'centroids_masks'), -1)
        image = crop_nonblack(image, mask, self.crop_size // 2)
        
        hr_image = normalize_img(image) if self.normalization else image     
        lr_image = downsize_img(image, self.upscale_factor)
        if self.colorisation:
            image = np.expand_dims(lr_image.astype(np.float).sum(-1), -1)
            lr_image = normalize_img(image) if self.normalization else image
        
        return ToTensor()(lr_image).float(), ToTensor()(hr_image).float()
    
    def __len__(self):
        return len(self.image_filenames)
        
class RecursionValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor, colorisation=False,
                 merged_channels=None, normalization=True):
        super().__init__()
        
        reference_table = pd.read_csv(dataset_dir + '.csv')
        self.image_filenames = []
        for (idx, row) in reference_table.iterrows():
            self.image_filenames += rec_mk_img_paths(row, dataset_dir)
    
        self.colorisation = colorisation
        self.normalization = normalization
        if merged_channels is not None:
            self.merged_channels = [_rx_channel_idxs[ch] for ch in merged_channels]
            self.not_merged_channels = [ch for ch in range(0, 6) if ch not in self.merged_channels]
        else:
            self.merged_channels = None
        
        self.upscale_factor = upscale_factor
        
    def __getitem__(self, index):
        image = np.stack([
            cv2.imread(img, -1) for img in self.image_filenames[index]
        ], axis=-1)
        if not self.colorisation:
            image = image.mean(-1).astype(np.uint8)
        elif self.merged_channels is not None:
            image = partial_channel_merge(image, self.merged_channels, self.not_merged_channels)
            
        image = normalize_img(image) if self.normalization else image
        w, h = image.shape[:2]
        
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
#         lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
#         hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = centr_crop(image, crop_size)
        lr_image = downsize_img(hr_image, self.upscale_factor)
        if self.colorisation:
            image = np.expand_dims(lr_image.astype(np.float).sum(-1), -1)
            lr_image = normalize_img(image) if self.normalization else image
        
        hr_restore_img = cv2.resize(
            lr_image, (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)
        
        if self.colorisation:
            ch_count = hr_image.shape[-1]
            hr_restore_img = np.stack([hr_restore_img]*ch_count, axis=2)
        
        return ToTensor()(lr_image).float(), ToTensor()(hr_restore_img).float(), ToTensor()(hr_image).float()
    
    def __len__(self):
        return len(self.image_filenames)

#
# Original SRGAN dataset instances
#

# class TrainDatasetFromFolder(Dataset):
#     def __init__(self, dataset_dir, crop_size, upscale_factor):
#         super(TrainDatasetFromFolder, self).__init__()
#         self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
#         crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
#         self.hr_transform = train_hr_transform(crop_size)
#         self.lr_transform = train_lr_transform(crop_size, upscale_factor)

#     def __getitem__(self, index):
#         hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
#         lr_image = self.lr_transform(hr_image)
#         return lr_image, hr_image

#     def __len__(self):
#         return len(self.image_filenames)


# class ValDatasetFromFolder(Dataset):
#     def __init__(self, dataset_dir, upscale_factor):
#         super(ValDatasetFromFolder, self).__init__()
#         self.upscale_factor = upscale_factor
#         self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

#     def __getitem__(self, index):
#         hr_image = Image.open(self.image_filenames[index])
#         w, h = hr_image.size
#         crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
#         lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
#         hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
#         hr_image = CenterCrop(crop_size)(hr_image)
#         lr_image = lr_scale(hr_image)
#         hr_restore_img = hr_scale(lr_image)
#         return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

#     def __len__(self):
#         return len(self.image_filenames)
    
    
# class TestDatasetFromFolder(Dataset):
#     def __init__(self, dataset_dir, upscale_factor):
#         super(TestDatasetFromFolder, self).__init__()
#         self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
#         self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
#         self.upscale_factor = upscale_factor
#         self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
#         self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

#     def __getitem__(self, index):
#         image_name = self.lr_filenames[index].split('/')[-1]
#         lr_image = Image.open(self.lr_filenames[index])
#         w, h = lr_image.size
#         hr_image = Image.open(self.hr_filenames[index])
#         hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
#         hr_restore_img = hr_scale(lr_image)
#         return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

#     def __len__(self):
#         return len(self.lr_filenames)