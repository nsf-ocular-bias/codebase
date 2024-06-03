# %%
from model import TrainableModel
from config import load_config
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin

from datasets.fairface import FairFaceDataModule
from datasets.utkface import UTKFaceDataModule
import os
from glob import glob
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# %%
import torch
weight = "F:\\Lab\\nfs\\nsl\\training-run\\lightning_logs\\version_104\\checkpoints\\epoch=58-val_std=2.0966.ckpt"

# weight = "F:\\Lab\\nfs\\nsl\\training-run\\lightning_logs\\version_122\\checkpoints\\epoch=42-val_std=1.9463.ckpt"


# %%
def init_directions(model):
    noises = []

    n_params = 0
    for name, param in model.named_parameters():
        delta = torch.normal(.0, 1., size=param.size()).to('cuda')
        nu = torch.normal(.0, 1., size=param.size()).to('cuda')

        param_norm = torch.norm(param)
        delta_norm = torch.norm(delta)
        nu_norm = torch.norm(nu)

        delta /= delta_norm
        delta *= param_norm

        nu /= nu_norm
        nu *= param_norm

        noises.append((delta, nu))

        n_params += np.prod(param.size())

    print(f'A total of {n_params:,} parameters.')

    return noises

# %%
def init_network(model, all_noises, alpha, beta):
    with torch.no_grad():
        for param, noises in zip(model.parameters(), all_noises):
            delta, nu = noises
            delta = delta.to('cuda')
            nu = nu.to('cuda')
            # the scaled noises added to the current filter
            new_value = param + alpha * delta + beta * nu
            param.copy_(new_value)
    return model

# %%
from datasets.fairface import FairFaceDataset

YAML_PATH = "F:\\Lab\\nfs\\nsl\\training-run\\lightning_logs\\version_104\\hparams.yaml"
# YAML_PATH = "F:\\Lab\\nfs\\nsl\\training-run\\lightning_logs\\version_122\\hparams.yaml"

import yaml
import json
from torchvision import transforms
from torch.utils.data import DataLoader
s = json.dumps(dict(yaml.load(open(YAML_PATH), Loader=yaml.Loader))['config'], default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)
with open('temp_eval_config.json', 'w') as f:
    f.write(s)
from config import load_config
cfg=load_config('temp_eval_config.json')


model = TrainableModel.load_from_checkpoint(weight, config=load_config('temp_eval_config.json')).to('cuda')
transform = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.Resize(224),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4824, 0.3578, 0.3045],
                                 std=[0.2571, 0.2242, 0.2182])
        ])

train_ds = FairFaceDataset(
                cfg, train=True, transform=transform, test_transform=transform, filter_by=None, split='test')
dataloader = DataLoader(train_ds, batch_size=cfg.eval.batch_size, pin_memory=True, num_workers=3, prefetch_factor=2, persistent_workers=True)



import matplotlib.pyplot as plt
# %%
# Creating the initial noise directions
import numpy as np
from tqdm import tqdm
def main():
    noises = init_directions(model)
    RESOLUTION = 50
    # Our loss function (for categorical problems)
    crit = torch.nn.CrossEntropyLoss()

    # The mesh-grid
    A, B = np.meshgrid(np.linspace(-1, 1, RESOLUTION),
                    np.linspace(-1, 1, RESOLUTION), indexing='ij')

    loss_surface = np.empty_like(A)


    for i in tqdm(range(RESOLUTION)):
        for j in range(RESOLUTION):
            total_loss = 0.
            n_batch = 0
            alpha = A[i, j]
            beta = B[i, j]
            # Initilazing the network to the current directions (alpha, beta)
            net = init_network(model, noises, alpha, beta).to('cuda')
            k = 0
            for images, labels, _ in tqdm(dataloader):
                images = images.to('cuda')
                labels = labels.to('cuda')
                # We do not net to acquire gradients
                with torch.no_grad():
                    preds = net(images)
                    preds = preds[1]
                    loss = crit(preds, labels)
                    total_loss += loss.item()
                    n_batch += 1
                k += 1
                if k > 2:
                    break
            loss_surface[i, j] = total_loss / (n_batch * cfg.eval.batch_size)
            # Freeing up GPU memory
            del net
            torch.cuda.empty_cache()


    BATCH_SIZE = cfg.eval.batch_size
    model_id = "baseline"
    plt.figure(figsize=(10, 10))
    plt.contour(A, B, loss_surface)
    plt.savefig(f'results/{model_id}_contour_bs_{BATCH_SIZE}_res_{RESOLUTION}_imagenetv2.png', dpi=100)
    plt.close()

    np.save(f'{model_id}_xx_imagenetv2.npy', A)
    np.save(f'{model_id}_yy_imagenetv2.npy', B)
    np.save(f'{model_id}_zz_imagenetv2.npy', loss_surface)


if __name__ == '__main__':
    main()