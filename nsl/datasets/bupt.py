import torch
from torchvision import transforms
import mxnet as mx
from mxnet import recordio


from .base import BaseDataModule
from config import datasets
from cutmix.cutmix import CutMix

from catalyst.data.sampler import DistributedSamplerWrapper
from torch.utils.data import DataLoader, random_split, Dataset, WeightedRandomSampler

from typing import Optional


class BUPTDataset(torch.utils.data.Dataset):
    def __init__(self, mxnet_record, mxnet_idx, config: dict, transform: Optional[transforms.Compose] = None,  test_transform: Optional[transforms.Compose] = None, train: bool = True):
        self.config = config
        self.transform = transform
        self.test_transform = test_transform
        self.train = train
        self.data = recordio.MXIndexedRecordIO(mxnet_idx, mxnet_record, 'r')

    def __len__(self):
        return len(self.data.keys) - 1

        #Add learning rate scaling

    def __getitem__(self, index):
        header, s = recordio.unpack(self.data.read_idx(index+1))
        image = mx.image.imdecode(s).asnumpy()
        label = int(header.label)
        if self.train:
            image = self.transform(image)
        else:
            image = self.test_transform(image)

        return image, torch.tensor(label, dtype=torch.long)


class BUPTDataModule(BaseDataModule):
    def __init__(self, config: dict):
        super().__init__(config)
        self.config = config
    
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
            self.train_ds = BUPTDataset(
                self.config, train=True, transform=self.transform, test_transform=self.test_transform, )
            self.val_ds = BUPTDataset(
                self.config, train=False, transform=self.transform, test_transform=self.test_transform,)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_ds = BUPTDataset(
                self.config, train=False, transform=self.transform, test_transform=self.test_transform, )

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
        return DataLoader(self.train_ds, batch_size=self.config.train.batch_size, pin_memory=True, num_workers=self.config.runtime.num_workers, prefetch_factor=4, persistent_workers=True, sampler=sampler, drop_last=True)