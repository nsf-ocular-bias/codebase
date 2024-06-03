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
from glob import glob
from turbojpeg import TurboJPEG

from typing import Optional
import os
import cv2

from .base import BaseDataModule
from config import datasets
from cutmix.cutmix import CutMix

from catalyst.data.sampler import DistributedSamplerWrapper

jpeg = TurboJPEG()

class UTKFaceDataset(Dataset):
    def __init__(self, config: dict, transform: Optional[transforms.Compose] = None,  test_transform: Optional[transforms.Compose] = None, train: bool = True, filter_by: Optional[str] = None, split: Optional[str] = None):
        super().__init__()
        self.config = config
        self.transform = transform
        self.test_transform = test_transform
        self.train = train
        self.split = split
        self.filter_by = filter_by
        self.data = glob(datasets["utkface"]["path"] + "/*")
        self.path = datasets["utkface"]["path"]

    def __len__(self):
        return len(self.data)

    def _to_rgb(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx):
        img_name = self.data[idx]

        in_file = open(img_name, 'rb')
        img = self._to_rgb(jpeg.decode(in_file.read()))
        in_file.close()
        if self.train:
            img = self.transform(img)
        else:
            img = self.test_transform(img)
    
        label = int(img_name.split("//")[-1].split("_")[1] == "0")
        race_gender_idx = 0
        race_idx = 0

        if self.config.model.multitask:
            label = [label, race_idx]

        return img, label, race_gender_idx


class UTKFaceDataModule(BaseDataModule):
    def __init__(self, config: dict, filter_by: Optional[str] = None):
        super().__init__(config)
        self.config = config
        self.filter_by = filter_by
        self.test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4824, 0.3578, 0.3045],
                                 std=[0.2571, 0.2242, 0.2182])
        ])

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
            self.train_ds = UTKFaceDataset(
                self.config, train=True, transform=self.transform, test_transform=self.test_transform, filter_by=self.filter_by, split='train')
            self.val_ds = UTKFaceDataset(
                self.config, train=False, transform=self.transform, test_transform=self.test_transform, filter_by=self.filter_by, split='val')

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_ds = UTKFaceDataset(
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
        return DataLoader(self.train_ds, batch_size=self.config.train.batch_size, pin_memory=True, num_workers=self.config.runtime.num_workers, prefetch_factor=2, persistent_workers=True, sampler=sampler, drop_last=True)




