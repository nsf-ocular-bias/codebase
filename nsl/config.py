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
from torchvision import models
from collections import defaultdict
import json
from types import SimpleNamespace


datasets = {
    "fairface": {
        "labels_train": r"F:\Lab\datasets\fairface\fairface_label_train.csv",
        "labels_val": r"F:\Lab\datasets\fairface\fairface_label_val.csv",
        "path": r"F:\Lab\datasets\fairface\fairface-img-margin025-trainval",
        "neighbor_url":  "file:F:/Lab/nfs/nsl/neighbors.tar"
    },
    "morph": {
        "labels_train": r"C:\Users\g728v562\Documents\datasets\Morph_Dataset\morph_all.csv",
        "labels_val": r"C:\Users\g728v562\Documents\datasets\Morph_Dataset\morph_all.csv",
        "path": r"C:\Users\g728v562\Documents\datasets\Morph_Dataset\All",
    },
    "diveface": {
        "labels_train": r"C:\Users\g728v562\Documents\datasets\DiveFace\diveface_all.csv",
        "labels_val": r"C:\Users\g728v562\Documents\datasets\DiveFace\diveface_all.csv",
        "path": r"C:\Users\g728v562\Documents\datasets\DiveFace\All",
    },
    "bupt": {
        "train": r"F:\Lab\datasets\BUPTFace\Equalizedface\train.rec",
        "index": r"F:\Lab\datasets\BUPTFace\Equalizedface\train.idx",
    },
    "utkface": {
        "path": r"F:\Lab\datasets\UTKFace",
    }
}


def load_config(config_file: str) -> SimpleNamespace:
    with open(config_file) as f:
        base_config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    return base_config


