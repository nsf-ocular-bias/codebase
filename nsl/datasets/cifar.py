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

import pytorch_lightning as pl
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from typing import Optional

from .base import BaseDataModule


class CIFARDataModule(BaseDataModule):
    def __init__(self, config: dict):
        super().__init__(config)
        self.config = config
        

    def prepare_data(self):
        '''
        Download and prepare the data here.
        '''
        # download
        CIFAR10(self.config.train.log_dir, train=True, download=True)
        CIFAR10(self.config.train.log_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        '''
        Prepare the model for this stage. This is called before the training.
        Process the data here.
        '''
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            full_ds = CIFAR10(
                self.config.train.log_dir, train=True, transform=self.transform)
            train_count = int(len(full_ds) * self.config.data.validation_split)
            self.train_ds, self.val_ds = random_split(
                full_ds, [train_count, len(full_ds) - train_count])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_ds = CIFAR10(
                self.config.train.log_dir, train=False, transform=self.transform)

    
    def teardown(self, stage: Optional[str] = None):
        # Used to clean-up when the run is finished
        ...
