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
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin

from datasets.fairface import FairFaceDataModule
from datasets.utkface import UTKFaceDataModule
import os
from glob import glob
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import click

@click.command()
@click.option('-v', help='Training run version to evaluate', required=True)
def main(v):
    config_path = r"temp_eval_config.json"

    PATH = r"F:\Lab\nfs\nsl\training-run\lightning_logs\version_{}\checkpoints\*".format(v)

    YAML_PATH = PATH.replace("checkpoints\\*", "hparams.yaml")

    import yaml
    import json
    s = json.dumps(dict(yaml.load(open(YAML_PATH), Loader=yaml.Loader))['config'], default=lambda o: o.__dict__, 
                sort_keys=True, indent=4)

    with open('temp_eval_config.json', 'w') as f:
        f.write(s)

    import torch
    import pandas as pd
    import numpy as np
    results = {}
    for f in glob(PATH):
    # for f in [r"F:\\Lab\\nfs\\nsl\\training-run\\lightning_logs\\version_108\\checkpoints\\epoch=93-val_std=1.7460.ckpt"]:
        result = []
        model = TrainableModel.load_from_checkpoint(f, config=load_config(config_path))

        trainer = pl.Trainer(
                devices=[1],
                accelerator="gpu",
                strategy=DDPPlugin(find_unused_parameters=False),
                precision=16,
                benchmark=True,
            )
        
        
        # data = UTKFaceDataModule(load_config(config_path), filter_by='All')
        # acc = trainer.test(model, data)
        # result.append(acc)

        accs = []
        # for race_gender in [None, 'East Asian-Male', 'Indian-Female', 'Black-Female', 'White-Male', 'Middle Eastern-Male', 'Latino_Hispanic-Male', 'Indian-Male', 'White-Female', 'Southeast Asian-Female', 'Southeast Asian-Male', 'Middle Eastern-Female', 'East Asian-Female', 'Latino_Hispanic-Female', 'Black-Male']:
        for race_gender in ['East Asian-Male', 'Indian-Female', 'Black-Female', 'White-Male', 'Middle Eastern-Male', 'Latino_Hispanic-Male', 'Indian-Male', 'White-Female', 'Southeast Asian-Female', 'Southeast Asian-Male', 'Middle Eastern-Female', 'East Asian-Female', 'Latino_Hispanic-Female', 'Black-Male']:
            data = FairFaceDataModule(load_config(config_path), filter_by=race_gender)
            acc = trainer.test(model, data)
            accs.append([race_gender, acc])


        acc = []
        total = 0
        crct = 0
        print(glob(r"F:\Lab\nfs\nsl\.temp-results\*.csv"))
        for g in glob(r"F:\Lab\nfs\nsl\.temp-results\*.csv"):
            df = pd.read_csv(g)
            race_gender = g.split("\\")[-1]
            acc.append(len(df[df.correct == 1]) / len(df))
            total += len(df)
            crct += len(df[df.correct == 1])

        import numpy as np
        print(np.array(acc).std())

        result.append([acc, np.array(acc).std(), crct / total])

        
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
        #     result.append({"threshold": i, "min_acceptance": min_, "std": std, "acc_mean": np.array(acc).mean()})
        #     result.append({"threshold": i, "acceptance": percent, "std": std, "acc": acc})

        results[f] = result
    print(results)

    with open("temp_results.json", "w") as f:
        json.dump(results, f)

# Calculate the PSNR for rejected samples as well as failed samples

if __name__ == "__main__":
    main()