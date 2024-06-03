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

import numpy as np
import pandas as pd
import torch
import click
from model import TrainableModel
from config import load_config
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from utils import Dirichlet_SOS

from datasets.fairface import FairFaceDataModule
import os
from tqdm import tqdm
from glob import glob
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

results = {}
from torchvision import transforms


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    transform = transforms.Compose(
                [
                    transforms.Normalize(mean=[0.4824, 0.3578, 0.3045],
                                         std=[0.2571, 0.2242, 0.2182])
                ]
            )
    # Adding clipping to maintain [0,1] range
    # perturbed_image = torch.clamp(perturbed_image, 0, 1)
    perturbed_image = transform(perturbed_image)
    # Return the perturbed image
    return perturbed_image


def test(model, device, test_loader, epsilon, config):

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target, _ in tqdm(test_loader):

        if config.model.multitask:
            target = target[0]
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output_ = model(data)
        
        output_ = output_[1]
        if config.model.evidential:
            alpha, output = torch.split(output_, 2, dim=-1)

        # get the index of the max log-probability
        init_pred = output.argmax()

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        if config.model.evidential: 
            loss = Dirichlet_SOS(output_, target)
        else:
            loss = F.nll_loss(output, target)


        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)
        output = output[1]
        if config.model.evidential:
            alpha, output = torch.split(output, 2, dim=-1)

        # Check for success
        # get the index of the max log-probability
        final_pred = output.argmax()
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append(
                    (init_pred.item(), final_pred.item(), adv_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append(
                    (init_pred.item(), final_pred.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon,
          correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


@click.command()
@click.option('-v', help='Training run version to evaluate', required=True)
def main(v):
    config_path = r"temp_eval_config.json"

    PATH = r"F:\Lab\nfs\nsl\training-run\lightning_logs\version_{}\checkpoints\*".format(
        v)

    YAML_PATH = PATH.replace("checkpoints\\*", "hparams.yaml")

    epsilons = [0, .05, .1, .15, .2, .25, .3]

    import yaml
    import json
    s = json.dumps(dict(yaml.load(open(YAML_PATH), Loader=yaml.Loader))['config'], default=lambda o: o.__dict__,
                   sort_keys=True, indent=4)

    with open('temp_eval_config.json', 'w') as f:
        f.write(s)
    for f in glob(PATH):
        result = []
        model = TrainableModel.load_from_checkpoint(
            f, config=load_config(config_path))

        model.to(torch.device("cuda"))
        model.eval()

        # trainer = pl.Trainer(
        #         devices=[0],
        #         accelerator="gpu",
        #         strategy=DDPPlugin(find_unused_parameters=False),
        #         precision=16,
        #         benchmark=True,
        #     )

        accs = []
        for race_gender in ['East Asian-Male', 'Indian-Female', 'Black-Female', 'White-Male', 'Middle Eastern-Male', 'Latino_Hispanic-Male', 'Indian-Male', 'White-Female', 'Southeast Asian-Female', 'Southeast Asian-Male', 'Middle Eastern-Female', 'East Asian-Female', 'Latino_Hispanic-Female', 'Black-Male']:
            config = load_config(config_path)
            config.eval.batch_size = 1
            data = FairFaceDataModule(config, filter_by=race_gender)
            data.setup(stage='test')
            dataloader = data.test_dataloader()
            test(model, "cuda", dataloader, epsilons[2], config)
            # acc = trainer.test(model, data)
            # accs.append([race_gender, acc])

        # acc = []
        # total = 0
        # crct = 0
        # for g in glob(r"F:\Lab\nfs\nsl\.temp-results\*.csv"):
        #     df = pd.read_csv(g)
        #     acc.append(len(df[df.correct == 1]) / len(df))
        #     total += len(df)
        #     crct += len(df[df.correct == 1])

        # import numpy as np
        # print(np.array(acc).std())

        # result.append([acc, np.array(acc).std(), crct / total])

        # for i in np.arange(0, 1, 0.05):
        #     acc = []
        #     percent = []
        #     for g in glob(r"F:\Lab\nfs\nsl\.temp-results\*.csv"):
        #         df = pd.read_csv(g)
        #         df_ = df[df.uncertainty < i]
        #         percent.append(len(df_) / len(df))
        #         if len(df_) == 0:
        #             acc.append(0)
        #         else:
        #             acc.append(len(df_[df_.correct == 1]) / len(df_))

        #     min_ = np.array(percent).min()
        #     std = np.array(acc).std()
        #     result.append([i, min_, std])

        # results[f] = result
    print(results)

    with open("temp_results.json", "w") as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()
