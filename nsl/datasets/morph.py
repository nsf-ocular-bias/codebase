# Copyright 2021 Sreeraj Ramachandran. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from numpy.lib.shape_base import split
import pytorch_lightning as pl
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import pandas as pd
import webdataset as wds

from turbojpeg import TurboJPEG

from typing import Optional
import os
import cv2

from .base import BaseDataModule
from config import datasets
from cutmix.cutmix import CutMix

from catalyst.data.sampler import DistributedSamplerWrapper
from sklearn.model_selection import train_test_split

jpeg = TurboJPEG()

class MorphDataset(Dataset):
    def __init__(self, config: dict, transform: Optional[transforms.Compose] = None,  test_transform: Optional[transforms.Compose] = None, train: bool = True, filter_by: Optional[str] = None, split: Optional[str] = None):
        super().__init__()
        self.config = config
        self.transform = transform
        self.test_transform = test_transform
        self.train = train
        self.split = split
        self.filter_by = filter_by
        validation_split = self.config.data.validation_split
        self.data = pd.read_csv(datasets['morph']['labels_train'])
        train, test = train_test_split(self.data, test_size=validation_split, random_state=42)
        if self.split == "train":
            self.data = train
        elif self.split == "test":
            self.data = test
        else:
            self.data = test

        
        self.path = datasets['morph']['path']

        # self.data = self.data[(self.data.age!='0-2') & (self.data.age!='3-9')]  # Filter out kids

        self.data["race_gender"] = self.data.race + "-" + self.data.gender
        self.data['race_gender_idx'] = pd.Categorical(self.data.race_gender)
        self.data.race_gender_idx = self.data.race_gender_idx.cat.codes

        self.data['race_idx'] = pd.Categorical(self.data.race)
        # self.data['age_idx'] = pd.Categorical(self.data.age)
        # self.data.age_idx = self.data.age_idx.cat.codes
        self.data.race_idx = self.data.race_idx.cat.codes

        if filter_by is not None:
            race, gender = filter_by.split("-")
            self.data = self.data[self.data.gender == gender]
            self.data = self.data[self.data.race == race]

    def __len__(self):
        return len(self.data)

    def _to_rgb(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx].file
        

        if self.config.model.nsl and self.train:
            img = []
            in_file = open(os.path.join(self.path,img_name), 'rb')
            img.append(self._to_rgb(jpeg.decode(in_file.read())))
            in_file.close()

            img_name = img_name.split("/")[-1].split(".")[0]
            for i in range(self.config.data.neighbor_size):
                fname = os.path.join(os.path.join(self.path, "neighbors"), img_name + "_" + str(i) + ".jpg")
                in_file = open(fname, 'rb')
                img.append(self._to_rgb(jpeg.decode(in_file.read()))) # Resize the image to 112x112
                in_file.close()
            img[1:] = [self.transform(img_) for img_ in img[1:]] # Apply the same transform to all images
            img[0] = self.transform(img[0]) 
    
        else:   
            img_name = os.path.join(self.path, img_name)
            in_file = open(img_name, 'rb')
            img = self._to_rgb(jpeg.decode(in_file.read()))
            in_file.close()
            if self.train:
                img = self.transform(img)
            else:
                img = self.test_transform(img)
        
        label = int(self.data.iloc[idx].gender == "Male")
        race_gender_idx = self.data.iloc[idx].race_gender_idx
        
        if self.config.model.multitask:
            label = [label, self.data.iloc[idx].race_idx]
            if self.config.model.multitask_age:
                label.append(self.data.iloc[idx].age_idx)

        return img, label, race_gender_idx


class MorphDataModule(BaseDataModule):
    def __init__(self, config: dict, filter_by: Optional[str] = None):
        super().__init__(config)
        self.config = config
        self.filter_by = filter_by

    def prepare_data(self):
        '''
        Download and prepare the data here.
        '''
        ...

    def setup(self, stage: Optional[str] = None):
        '''
        Prepare the model for this stage. This is called before the training.
        Process the data here.
        '''
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_ds = MorphDataset(
                self.config, train=True, transform=self.transform, test_transform=self.test_transform, filter_by=self.filter_by, split='train')
            self.val_ds = MorphDataset(
                self.config, train=False, transform=self.transform, test_transform=self.test_transform, filter_by=self.filter_by, split='val')

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_ds = MorphDataset(
                self.config, train=False, transform=self.transform, test_transform=self.test_transform, filter_by=self.filter_by, split='test')

    def teardown(self, stage: Optional[str] = None):
        # Used to clean-up when the run is finished
        ...

    def train_dataloader(self) -> DataLoader:
        # TODO: Add mixup
        sampler = WeightedRandomSampler([1] * len(self.train_ds), num_samples=len(self.train_ds))
        sampler = DistributedSamplerWrapper(sampler)
        if self.config.data.cutmix_prob > 0:
            self.train_ds = CutMix(self.train_ds, num_class=self.config.data.num_classes,
                                   beta=self.config.data.cutmix_beta, prob=self.config.data.cutmix_prob)
        return DataLoader(self.train_ds, batch_size=self.config.train.batch_size, pin_memory=True, num_workers=self.config.runtime.num_workers, prefetch_factor=2, persistent_workers=True, sampler=sampler)




