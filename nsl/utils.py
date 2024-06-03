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
import torch
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader, random_split, Dataset, WeightedRandomSampler
from catalyst.data.sampler import DistributedSamplerWrapper
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from datasets.base import BaseDataModule
from datasets.fairface import FairFaceDataModule
from datasets.bupt import BUPTDataModule
from datasets.utkface import UTKFaceDataModule
from datasets.morph import MorphDataModule
from datasets.diveface import DiveFaceDataModule

def get_dataset(config) -> BaseDataModule:
    if config.data.ds_name == "fairface" and config.model.nsl:
        return FairFaceDataModule
    elif config.data.ds_name == "fairface" and not config.model.nsl:
        return FairFaceDataModule
    elif config.data.ds_name == "bupt":
        return BUPTDataModule
    elif config.data.ds_name == "utkface":
        return UTKFaceDataModule
    elif config.data.ds_name == "morph":
        return MorphDataModule
    elif config.data.ds_name == "diveface":
        return DiveFaceDataModule
    else:
        raise ValueError("Invalid dataset : {}".format(config.data.ds_name))


def get_lr_scheduler(sched, config, optimizer):
    # TODO add warmup scheduler
    if sched == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.train.num_epochs, eta_min=config.train.lr_min)
    elif sched == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.train.lr_step_size, gamma=config.train.lr_decay_factor)
    elif sched == "exponential":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=config.train.lr_decay_factor)
    elif sched == "reduceonplateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, verbose=True)
    elif sched == "constant":
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda _: 1)
    else:
        raise ValueError("Invalid scheduler : {}".format(sched))

    return lr_scheduler


class DataReweightingCallback(Callback):

    def _calculate_weights(self, pl_module):
        weights = []
        for race_gender in range(14):
            a = np.array([race_gender] * len(self.z)) == np.array(self.z)

            outputs = self.outputs[a]
            y = self.y[a]
            if pl_module.config.model.evidential:
                weight = outputs.mean()
                weight = pl_module.loss_ce(outputs, y.type(torch.LongTensor))
            else:
                weight = pl_module.loss(outputs, y.type(torch.LongTensor))
            weights.append(weight)

        weights = torch.stack(weights)
        weights = weights/weights.mean()  # Or median
        return weights

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        if pl_module.config.model.evidential:
            self.outputs.extend(
                outputs['logits'].type(torch.FloatTensor))
            # logger = trainer.logger
            # tensorboard = logger.experiment
            # logits = outputs['logits'].cpu().type(torch.FloatTensor)
            # alpha, _ = torch.split(logits, 2, dim=-1)
            # tensorboard.add_histogram('alpha', alpha.mean(dim=0), trainer.global_step)
        else:
            self.outputs.extend(
                outputs['logits'].type(torch.FloatTensor))
        x, y, z = batch
        if pl_module.config.model.multitask:
            y = y[0]
        self.z.extend(z)
        self.y.extend(y)

    def on_train_epoch_end(self, trainer, pl_module):

        self.outputs = torch.stack(self.outputs)
        self.y = torch.Tensor(self.y)
        self.z = torch.Tensor(self.z)
        weights = self._calculate_weights(pl_module)
        print("Reweighting weights : {}".format(weights))

        train_dataloader = trainer.train_dataloader

        if train_dataloader is not None:
            data = train_dataloader.dataset.datasets.data
            weights = weights[data.race_gender_idx.tolist()]
            sampler = train_dataloader.sampler
            sampler.sampler.weights = weights

    def on_train_epoch_start(self, trainer, pl_module):
        self.outputs = []
        self.z = []
        self.y = []


# Evidential Layers and Losses
class DenseDirichlet(nn.Module):
    def __init__(self, in_features, units, evidence='exp'):
        """
        A dense layer with Dirichlet prior.
        :param units: number of units in the dense layer
        """
        super(DenseDirichlet, self).__init__()
        self.units = int(units)
        self.dense = nn.Linear(in_features, self.units)
        self.evidence = evidence

    def forward(self, x):
        output = self.dense(x)
        if self.evidence == 'exp':
            evidence = torch.exp(torch.clamp(output, -10, 10)) # Exponential evidence
        elif self.evidence == 'softplus':
            evidence = torch.softplus(output) # Softplus evidence
        elif self.evidence == "relu":
            evidence = torch.relu(output)   # ReLU evidence
        alpha = evidence + 1
        prob = alpha / torch.sum(alpha, 1, keepdim=True)
        return torch.cat([alpha, prob], dim=-1)


def Dirichlet_SOS(alpha, y):
    def KL(alpha):
        beta = torch.ones((1, alpha.shape[1])).type(
            torch.FloatTensor).to(alpha.device)
        S_alpha = torch.sum(alpha, axis=1, keepdim=True)
        S_beta = torch.sum(beta, axis=1, keepdim=True)
        lnB = torch.lgamma(S_alpha) - \
            torch.sum(torch.lgamma(alpha), axis=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), axis=1,
                            keepdim=True) - torch.lgamma(S_beta)
        lnB_uni = torch.sum(torch.lgamma(beta), axis=1,
                            keepdim=True) - torch.lgamma(S_beta)

        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)

        kl = torch.sum((alpha - beta)*(dg1-dg0), axis=1,
                       keepdim=True) + lnB + lnB_uni
        return kl

    alpha, prob = torch.split(alpha, alpha.shape[1]//2, dim=-1)


    y = F.one_hot(y.to(torch.int64), num_classes=alpha.shape[1]).type(
        torch.FloatTensor).to(alpha.device)
    S = torch.sum(alpha, axis=1, keepdim=True)
    evidence = alpha - 1
    m = alpha / S
    A = torch.sum((y-m)**2, axis=1, keepdim=True)
    B = torch.sum(alpha*(S-alpha)/(S*S*(S+1)), axis=1, keepdim=True)

    alpha_hat = y + (1-y)*alpha
    C = KL(alpha_hat)

    C = torch.mean(C, dim=1)
    return torch.mean(A + B + C)




# Calculate Jensen-Shannon Divergence
def jsdiv(p, q):
    m = 0.5 * (F.softmax(p, dim=1) + F.softmax(q, dim=1))
    p = torch.log_softmax(p, dim=1)
    q = torch.log_softmax(q, dim=1)
    return 0.5 * (F.kl_div(p, m, reduction='batchmean') + F.kl_div(q, m, reduction='batchmean'))