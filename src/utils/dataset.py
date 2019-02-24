import os
import cv2
import numpy as np
import pandas as pd

from ..configs import config

import sklearn.model_selection

import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Normalize, Compose


MEAN = np.array([0.03831719, 0.07311399, 0.05059095, 0.07424918])
STD = np.array([0.07944151, 0.13024841, 0.14401576, 0.12440699])

img_transform = Compose([
    ToTensor(),
    Normalize(mean=MEAN, std=STD)
])


def get_folds(n_splits, random_state=42):
    kfolds = sklearn.model_selection.StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    ksplit = kfolds.split(
        config.labels[config.labels.Type.isin([0, 1])].reset_index(drop=True),
        config.labels[config.labels.Type.isin([0, 1])].reset_index(drop=True).Target.apply(
            lambda x: sorted(x, key=lambda k: config.reverse_appearance[k])[0]
        )
    )
    return list(ksplit)


def get_fold_split(folds, fold):
    return folds[fold][0], folds[fold][1]


def get_datasets(folds, fold, augmentations=None):
    train, valid = get_fold_split(folds, fold)
    train_dataset = ProteinDataset(
        pd.concat([
            config.labels[config.labels.Type.isin([0, 1])].reset_index(drop=True).loc[train],
            config.labels[config.labels.Type == 2]
        ]).reset_index(drop=True), 
        config.PATHS['TRAIN'], 
        config.label_names_list, 
        augmentations=augmentations
    )
    valid_dataset = ProteinDataset(
        config.labels[config.labels.Type.isin([0, 1])].reset_index(drop=True).loc[valid], 
        config.PATHS['TRAIN'], 
        config.label_names_list, 
        augmentations=None
    )
    return train_dataset, valid_dataset


def get_datagens(train_dataset, valid_dataset):
    esmplr = EqualizedSampler(
        train_dataset,
        support=config.PARAMS['SUPPORT_CLASS_AMOUNT'],
        coeff=config.PARAMS['SUPPORT_POWER']
    )
    vsmplr = ValSampler(valid_dataset)

    train_datagen = torch.utils.data.DataLoader(
        train_dataset, 
        pin_memory=True,
        sampler=esmplr,
        batch_size=config.PARAMS['BATCH_SIZE'],
        num_workers=4
    )
    valid_datagen = torch.utils.data.DataLoader(
        valid_dataset,
        pin_memory=True,
        sampler=vsmplr,
        batch_size=config.PARAMS['BATCH_SIZE'],
        num_workers=2
    )

    return train_datagen, valid_datagen


class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, labels, basepath, label_names, augmentations=None):
        self.transform = augmentations
        self.basepath = basepath
        self.labels = labels.reset_index(drop=True)
        self.targets = self.labels.Target
        self.keys = self.labels.Id
        self.labels = self.labels[[key for key in label_names]]

    def __getitem__(self, idx):
        image = self.load_image(idx)
        mask = self.load_mask(idx)
        label = self.load_label(idx)

        data = { "image": image, 'instances': mask }
        if self.transform is not None:
            augmented = self.transform(data)
            image = augmented["image"]

        return self.postprocess(idx, image, label, mask)

    def postprocess(self, idx, image, label, mask):
        return { 
            'image': img_transform(image.astype(np.uint8)),
            'pid': self.keys[idx],
            'label': label.astype(np.float32),
#             'mask': mask,
        }

    def load_image(self, idx):
        postfixs = ['_blue_red_green', '_yellow']
        image_id = self.keys[idx]
        basepath = os.path.join(self.basepath, image_id)

        images = [
            cv2.imread(''.join([basepath, pstfx, '.png'])) 
            if pstfx == '_blue_red_green' else
            np.expand_dims(cv2.imread(''.join([basepath, pstfx, '.png']) , 0), -1)
            for pstfx in postfixs
        ]
        images = np.dstack(images)
        return images

    def load_mask(self, idx):
        image_id = self.keys[idx]
        basepath = os.path.join(self.basepath, image_id)

        mask = cv2.imread(''.join([basepath, '_mask', '.png']), -1)
        return mask

    def load_label(self, idx):
        return self.labels.loc[idx].values

    def __len__(self):
        return len(self.keys)


class EqualizedSampler(Sampler):
    r"""Samples elements randomly, without replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, dataset, support=700, coeff=.3):
        query = dataset.targets.apply(lambda x: sorted(x, key=lambda k: config.reverse_appearance[k])[0])
        self.groups = query.groupby(query).groups.values()
        self.groups = [np.array(v) for v in self.groups]
        self.amounts = np.array([len(v) for v in self.groups])
        self.amounts = self.amounts * np.power(support / self.amounts, coeff)
        self.amounts = self.amounts.astype(np.int)

    def __iter__(self):
        for v in self.groups:
            np.random.shuffle(v)
        idxs = np.concatenate([np.random.choice(v, self.amounts[i]) for i, v in enumerate(self.groups)])
        np.random.shuffle(idxs)
        return iter(idxs)

    def __len__(self):
        return sum(self.amounts)


class ValSampler(Sampler):
    r"""Samples elements randomly, without replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, dataset):
        self.idxs = list(dataset.labels.index)

    def __iter__(self):
        return iter(self.idxs)

    def __len__(self):
        return len(self.idxs)
