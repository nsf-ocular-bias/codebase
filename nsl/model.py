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

import os

from click import style

import utils

import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy
import timm
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import kornia
import clip

from utils import DenseDirichlet, Dirichlet_SOS, jsdiv
import sys
sys.path.append(r'F:\Lab\nfs\nsl\BLIP\\')
from models.blip import blip_feature_extractor

class TrainableModel(LightningModule):
    def __init__(self, config: dict):
        '''
        Create a model with the given configuration.
        :param config:
        '''
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        if self.config.model.use_clip_encoder:
            self.base_model = clip.load("ViT-L/14")[0].visual
            self.base_model.proj = nn.Parameter(torch.eye(1024)) #.cuda().half())
            self.ll = nn.Linear(1024, 1024)
            self.act = nn.ReLU()
            for param in self.base_model.parameters():
                param.requires_grad = False
        elif self.config.model.use_blip_encoder:
            model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth'
            self.base_model = blip_feature_extractor(pretrained=model_url, image_size=224, vit='large')
            for param in self.base_model.parameters():
                param.requires_grad = False
            self.ll = nn.Linear(1024, 1024)
            self.act = nn.ReLU()
        else:
            self.base_model = timm.create_model(
                config.model.model_name, pretrained=True)

        self.accuracy = Accuracy()
        if self.config.model.nsl_distance == "l2":
            self.pdist = nn.PairwiseDistance(p=2)
        elif self.config.model.nsl_distance == "cosine":
            self.pdist = nn.CosineSimilarity(dim=1)
        elif self.config.model.nsl_distance == "jsdiv":
            self.pdist = jsdiv
        elif self.config.model.nsl_distance == "kldiv":
            self.pdist = kornia.losses.kl_div_loss_2d
        else:
            raise ValueError("Unknown distance type")
        self.loss_ce = nn.CrossEntropyLoss(label_smoothing=config.train.label_smoothing)
        feat_dim = 1024 if (self.config.model.use_clip_encoder or self.config.model.use_blip_encoder) else self.base_model.classifier.out_features
        if self.config.model.evidential:
            self.loss = Dirichlet_SOS
            self.fc = DenseDirichlet(feat_dim, self.config.model.num_classes, evidence='exp')
        else:
            self.loss = nn.CrossEntropyLoss(
                label_smoothing=config.train.label_smoothing)
            self.fc = nn.Linear(
                feat_dim, config.model.num_classes)
        
        if self.config.model.multitask:
            self.fc_race = nn.Linear(feat_dim, config.model.num_race_classes)
            if self.config.model.multitask_age:
                self.fc_age = nn.Linear(feat_dim, config.model.num_age_classes)
        self.current_labels_y = []
        self.current_labels_r = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the model.
        '''
        if self.config.model.use_clip_encoder:
            x = self.base_model(x)
            x = self.act(self.ll(x))
        elif self.config.model.use_blip_encoder:
            x = self.base_model(x, caption="", mode='image')[:,0,:]
            x = self.act(self.ll(x))
        else:
            x = self.base_model(x)
        if self.config.model.multitask:
            out =  [x, self.fc(x), self.fc_race(x)]
            if self.config.model.multitask_age:
                out.append(self.fc_age(x))
            return out
        else:
            return [x, self.fc(x)]

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        x, y, _ = batch

        if self.config.model.nsl:
            neighbors = torch.cat(x[1:])
            x = x[0]
            
        preds = self(x) # Change the weight of generated images

    

        if self.config.model.multitask:
            if self.config.model.multitask_age:
                sample_embeddings, preds, preds_race, preds_age = preds
            else:
                sample_embeddings, preds, preds_race = preds
            loss = self.loss(preds, y[0]) + self.loss_ce(preds_race, y[1].type(torch.int64)) # Fix aware and unaware, Add uncertainty to the loss, Add evidence less loss as well. as another task
            if self.config.model.multitask_age:
                loss += self.loss_ce(preds_age, y[2].type(torch.int64))
            # Get evidence from two classifiers and then compare
        else:
            sample_embeddings, preds = preds
            loss = self.loss(preds, y)
        if self.config.model.nsl and batch_idx % self.config.model.lazy_regularization == 0: # Lazy regularization
            self.eval()
            with torch.no_grad():
                if self.config.model.use_clip_encoder:
                    neighbor_embeddings = self.act(self.ll(self.base_model(neighbors)))
                elif self.config.model.use_blip_encoder:
                    neighbor_embeddings = self.act(self.ll(self.base_model(neighbors, caption="", mode='image')[:,0,:]))
                else:
                    neighbor_embeddings = self.base_model(neighbors)
            self.train()
            sample_embeddings = sample_embeddings.repeat_interleave(self.config.data.neighbor_size, dim=0)
            neighbor_loss = self.config.model.nsl_neighbor_weight * self.pdist(neighbor_embeddings, sample_embeddings)
            self.log("neighbor_loss", neighbor_loss.mean(), on_step=True,
                on_epoch=True, prog_bar=True, logger=True, batch_size=self.config.train.batch_size)
            loss += neighbor_loss.mean()
        if self.config.model.evidential:
            alpha, logits = torch.split(preds, 2, dim=-1)
            uncertainty =  alpha.shape[1] / torch.sum(alpha, 1, keepdim=True)
            # loss += uncertainty.mean()
        self.log("train_loss", loss, on_step=False,
                on_epoch=True, prog_bar=True, logger=True, batch_size=self.config.train.batch_size)
        if self.config.model.evidential:
            return {'loss': loss, 'logits': preds.detach(), 'uncertainty': uncertainty.detach()}
        else:
            return {"loss": loss, 'logits': preds.detach()}

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        x, y, z = batch
        
        logits = self(x)
        feats = logits[0]
        if self.config.model.multitask:
            loss = self.loss(logits[1], y[0]) #+ self.loss_ce(logits[2], y[1].type(torch.int64)) # Fix aware and unaware
            if self.config.model.multitask_age:
                loss += self.loss_ce(logits[3], y[2].type(torch.int64))
            y = y[0]
        else:
            loss = self.loss(logits[1], y)
        logits = logits[1]
        if self.config.model.evidential:
            alpha, logits = torch.split(logits, 2, dim=-1)
            uncertainty =  alpha.shape[1] / torch.sum(alpha, 1, keepdim=True)
        correct = (logits.argmax(1) == y).float()
            # loss += uncertainty.mean()
        # preds = torch.argmax(logits, dim=1)
        # acc = self.accuracy(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        # self.log("val_acc", acc, prog_bar=True)
        if self.config.model.evidential:
            self.log("uncertainty", uncertainty, prog_bar=True)
            return {'loss': loss, 'logits': logits, 'uncertainty': uncertainty, "correct": correct, "z": z, "feats": feats}
        else:
            return {'loss': loss, 'logits': logits, "correct": correct, "z": z, "feats": feats}

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        return self.validation_step(batch, batch_idx)
    
    def test_epoch_end(self, outputs):
        outputs = self.all_gather(outputs)
        filter_by = self.trainer.test_dataloaders[0].dataset.filter_by
        if filter_by is None:
            z = torch.cat([x['z'].flatten() for x in outputs]).cpu().numpy()
            feats = torch.cat([x['feats'] for x in outputs], dim=1).squeeze().cpu().numpy()
            X_embedded = TSNE(n_components=2, init="random").fit_transform(feats)
            plt.figure(figsize=(16,10))
            tsne_plt = sns.scatterplot(
                x=X_embedded[:,0], y=X_embedded[:,1],
                hue=z,
                palette=sns.color_palette(n_colors=14) ,
                legend="full",
                # alpha=0.3
                # style=z
            )
            plt.legend(title='t-SNE', loc='upper left', labels=['East Asian-Male', 'Indian-Female', 'Black-Female', 'White-Male', 'Middle Eastern-Male', 'Latino_Hispanic-Male', 'Indian-Male', 'White-Female', 'Southeast Asian-Female', 'Southeast Asian-Male', 'Middle Eastern-Female', 'East Asian-Female', 'Latino_Hispanic-Female', 'Black-Male'])
            fig = tsne_plt.get_figure()
            fig.savefig("tsne.png") 
        else:
            if self.config.model.evidential:
                correct = torch.cat([x['correct'].flatten() for x in outputs]).cpu().numpy()
                uncertainty = torch.cat([x['uncertainty'].flatten() for x in outputs]).cpu().numpy()
                

                df = pd.DataFrame({"correct": correct, "uncertainty": uncertainty})
                df.to_csv(".temp-results\\uncertainty_" + filter_by + ".csv")
            else:
                correct = torch.cat([x['correct'].flatten() for x in outputs]).cpu().numpy()
                # uncertainty = torch.cat([x['uncertainty'].flatten() for x in outputs]).cpu().numpy()
                df = pd.DataFrame({"correct": correct})
                df.to_csv(".temp-results\\uncertainty_" + filter_by + ".csv")
        
        return outputs
    
    def validation_epoch_end(self, outputs):
        outputs = self.all_gather(outputs)
        correct = torch.cat([x['correct'].flatten() for x in outputs])
        zs = torch.cat([x['z'].flatten() for x in outputs])
        acc = torch.count_nonzero(correct) / correct.shape[0]
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, rank_zero_only=True)
        accs = []
        for z in torch.unique(zs):
            accs.append(torch.count_nonzero(correct[zs == z]) / (zs == z).sum())
        self.log("val_std", torch.Tensor(accs).std() * 100, on_epoch=True, prog_bar=True, rank_zero_only=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        opt = self.config.train.optimizer
        sched = self.config.train.lr_sched

        if opt == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.config.train.lr_base, weight_decay=self.config.train.weight_decay)
        elif opt == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.config.train.lr_base, weight_decay=self.config.train.weight_decay)
        elif opt == "momentum":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.config.train.lr_base, weight_decay=self.config.train.weight_decay, momentum=self.config.train.momentum)
        elif opt == "rmsprop":
            optimizer = torch.optim.RMSprop(
                self.parameters(), lr=self.config.train.lr_base, weight_decay=self.config.train.weight_decay, eps=0.001, alpha=0.9)
        else:
            raise ValueError("Invalid optimizer : {}".format(opt))

        lr_scheduler = utils.get_lr_scheduler(sched, self.config, optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss",
            },
        }
