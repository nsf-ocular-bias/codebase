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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, Sampler
from torchvision import transforms
from cutmix.cutmix import CutMix

from typing import Optional


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.test_transform = transforms.Compose([
            # transforms.GaussianBlur(kernel_size=3),
            # transforms.RandomAdjustSharpness(1.0),
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4824, 0.3578, 0.3045],
                                 std=[0.2571, 0.2242, 0.2182])
        ])
        if config.model.use_clip_encoder or config.model.use_blip_encoder:
            self.test_transform = transforms.Compose(
                    [transforms.ToPILImage(),
                        transforms.Resize(224),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
                    ]
                )

        if config.data.augmentation == "default":
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4824, 0.3578, 0.3045],
                                         std=[0.2571, 0.2242, 0.2182])
                ]
            )
        elif config.model.use_clip_encoder or config.model.use_blip_encoder:
            self.transform = transforms.Compose(
                [transforms.ToPILImage(),
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
                ]
            )
        elif config.data.augmentation == "randaugment":
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(224), # Preprocess the resize
                    transforms.CenterCrop(224),
                    transforms.RandAugment(
                        config.data.randaugment_n, config.data.randaugment_m),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4824, 0.3578, 0.3045],
                                         std=[0.2571, 0.2242, 0.2182])
                ]
            )
        elif config.data.augmentation == "autoaugment":
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    # transforms.Resize(224),
                    transforms.AutoAugment(),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4824, 0.3578, 0.3045],
                                         std=[0.2571, 0.2242, 0.2182])
                ]
            )
        else:
            raise ValueError(
                f"Unknown augmentation type: {config.data.augmentation}")

    def train_dataloader(self) -> DataLoader:
        # TODO: Add mixup
        if self.config.data.cutmix_prob > 0:
            self.train_ds = CutMix(self.train_ds, num_class=self.config.data.num_classes,
                                   beta=self.config.data.cutmix_beta, prob=self.config.data.cutmix_prob)
        return DataLoader(self.train_ds, batch_size=self.config.train.batch_size, pin_memory=True, num_workers=self.config.runtime.num_workers, prefetch_factor=2, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.config.eval.batch_size, pin_memory=True,  num_workers=self.config.runtime.num_workers, prefetch_factor=2, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=self.config.eval.batch_size)

