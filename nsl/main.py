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

from model import TrainableModel
from config import load_config
from utils import get_dataset, DataReweightingCallback




import click
import logging
from pprint import pformat
import os
import ssl

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:4096"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging, BackboneFinetuning
import torch

ssl._create_default_https_context = ssl._create_unverified_context


# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pytorch_lightning")


# GPU selection
AVAIL_GPUS = min(1, torch.cuda.device_count())


def train(config) -> None:
    '''
    Function to train the model.
    '''
    model = TrainableModel(config)

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=200, verbose=False, mode="min")
    checkpoint_callback = ModelCheckpoint(save_top_k=10, monitor="val_std", save_last=True, filename='{epoch}-{val_std:.4f}', every_n_train_steps=None,)
    swa_callback = StochasticWeightAveraging(swa_epoch_start=25)

    data_reweighting_callback = DataReweightingCallback()

    multiplicative = lambda epoch: 1.5
    finetuning_callback = BackboneFinetuning(config.train.num_epochs // 4, multiplicative, verbose=True)

    callbacks = [checkpoint_callback]
    if config.train.swa:
        callbacks.append(swa_callback)
    if config.train.data_reweighting:
        callbacks.append(data_reweighting_callback)
    if config.model.finetuning:
        callbacks.append(finetuning_callback)

    # Select the dataset
    data = get_dataset(config)(config)

    trainer = pl.Trainer(
        devices=[0, 1],
        accelerator="gpu",
        max_epochs=config.train.num_epochs,
        strategy=DDPPlugin(find_unused_parameters=False),
        precision=16,
        callbacks=callbacks,
        benchmark=True,
        default_root_dir=config.train.log_dir,
        replace_sampler_ddp=True,

        # detect_anomaly=True,

        # profiler="simple",
        # limit_train_batches=5,
        # limit_val_batches=5,
    )

    # start training
    trainer.fit(model, data, ckpt_path=config.model.resume)


def test(config: dict) -> None:
    '''
    Function to test the model.
    '''
    model = TrainableModel.load_from_checkpoint(config.model.resume, config=config)


    data = get_dataset(config.data.ds_name)(config)

    trainer = pl.Trainer(
        devices=[0, 1],
        accelerator="gpu",
        max_epochs=config.train.num_epochs,
        strategy=DDPPlugin(find_unused_parameters=False),
        precision=16,
        benchmark=True,
        default_root_dir=config.train.log_dir,
        # profiler="advanced"
    )

    trainer.test(model, data)


@click.command()
@click.option('--config', default=r"F:\Lab\nfs\nsl\configs\base_config.json", help='Configuration file.')
@click.option('--mode', default="train", help='Mode to run the script in.')
def main(config: str, mode: str) -> None:
    pl.seed_everything(42, workers=True)
    base_config = load_config(config)
    log.info(pformat(base_config))

    if mode == 'train':
        train(base_config)
    elif mode == "test":
        test(base_config)


if __name__ == '__main__':
    main()
