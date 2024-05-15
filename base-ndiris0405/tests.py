# Weights and Biases Test

import wandb
api = wandb.Api()


from glob import glob
wandb.tensorboard.patch(tensorboardX=False)

for f in glob("logs/*"):
    api.sync_tensorboard(f)