import torch
import torch.nn.functional as F
import os
import json

with open("evaluate\\weights.json", "r") as f:
    weights = json.load(f)

import click
import pandas as pd
import numpy as np
path = r"F:\Lab\datasets\fairface\fairface-img-margin025-trainval\\"

def jsdiv(p, q):
    m = 0.5 * (F.softmax(p) + F.softmax(q))
    p = torch.log_softmax(p, dim=0)
    q = torch.log_softmax(q, dim=0)
    return 0.5 * (F.kl_div(p, m, reduction='batchmean') + F.kl_div(q, m, reduction='batchmean'))


@click.command()
@click.option('-v', help='Model to evaluate', required=True)
def main(v):
    print("Loading weights {}".format(weights[v]))
    ckpt_path = weights[v]
    YAML_PATH = '\\'.join(ckpt_path.split("\\")[0:-2]) + "\\hparams.yaml"
    print(YAML_PATH)

    import yaml
    config = dict(yaml.load(open(YAML_PATH), Loader=yaml.Loader))['config']

    import cv2
    import sys
    sys.path.append(r"F:\Lab\nfs\nsl")
    from model import TrainableModel

    from glob import glob
    from tqdm import tqdm
    from torchvision import transforms
    import dlib
    from turbojpeg import TurboJPEG
    # from utils import jsdiv
    # detector = dlib.get_frontal_face_detector()
    # sp = dlib.shape_predictor(r"C:\Users\g728v562\Downloads\shape_predictor_5_face_landmarks.dat\shape_predictor_5_face_landmarks.dat")
    jpeg = TurboJPEG()
    device = 'cuda'
    with_dlib = False
    batch_size = 128
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4824, 0.3578, 0.3045],
                             std=[0.2571, 0.2242, 0.2182])
    ])

    model = TrainableModel.load_from_checkpoint(
        ckpt_path, config=config).to(device)
    model.eval()

    df = pd.read_csv(r"F:\Lab\datasets\fairface\fairface_label_val.csv")
    # from sklearn.model_selection import train_test_split
    # _, df = train_test_split(df, test_size=0.2, random_state=42)
    overall_y_pred = []
    overall_y_true = []
    accs = []
    # for RACE in df.race.unique():
    #     df_ = df[df.race == RACE]
    #     for GENDER in ["Male", "Female"]:
    #         df_n = df_[df_.gender == GENDER]
    #         images = []

    #         y_pred = []
    #         y_true = []
    #         us = []

    #         # paths = glob(path + RACE + "\\" + GENDER + "\\*")
    #         paths = list(df_n.file)
    #         paths = [os.path.join(r"F:\Lab\datasets\fairface\fairface-img-margin025-trainval", x) for x in paths]

    #         length = len(paths)
    #         for i, img in enumerate(tqdm(paths)):
    #             in_file = open(img, 'rb')
    #             images.append(cv2.cvtColor(jpeg.decode(in_file.read()), cv2.COLOR_BGR2RGB))
    #             label = int(GENDER == "Male")
    #             y_true.append(label)
    #             overall_y_true.append(label)
    #             in_file.close()

    #             # dets = detector(img, 1)
    #             # faces = dlib.full_object_detections()

    #             # for detection in dets:
    #             #     faces.append(sp(img, detection))

    #             if len(images) == batch_size or i == length - 1:
    #                 images = torch.stack([transform(image) for image in images])
    #                 images = images.to(device)
    #                 with torch.no_grad():
    #                     outputs = model(images)
    #                     outputs = outputs[1]
    #                     if "edl" in v:
    #                         alpha, outputs = torch.split(outputs, 2, dim=-1)
    #                         uncertainty =  alpha.shape[1] / torch.sum(alpha, 1, keepdim=True)
    #                         us.extend(uncertainty)

    #                     y_pred.extend(outputs)
    #                     overall_y_pred.extend(outputs)
    #                 images = []

    #         y_pred = torch.stack(y_pred).cpu()
    #         y_true = torch.tensor(y_true).cpu()
    #         # us = torch.stack(us).cpu().flatten()
    #         from torchmetrics.functional import accuracy,confusion_matrix

    #         acc = accuracy(y_pred.argmax(1), y_true)

    #         print(RACE, GENDER, acc)
    #         # acc = accuracy(y_pred.argmax(1)[us < 0.03], y_true[us < 0.03])

    #         # print(RACE, GENDER, acc, len(y_true[us < 0.03]))
    #         accs.append(acc)

    # print(np.std(np.array(accs)) * 100)

    # y_pred = torch.stack(overall_y_pred).cpu()
    # y_true = torch.tensor(overall_y_true).cpu()
    # # print(y_pred.shape, y_true.shape)
    # acc = accuracy(y_pred.argmax(1), y_true)
    # print(acc)

    jsdivs = []
    jsdivs_dict = {}
    # Calculate JSDIV on all races
    for RACE in df.race.unique():
        df_ = df[df.race == RACE]
        features = {"Male": [], "Female": []}
        for GENDER in ["Male", "Female"]:
            df_n = df_[df_.gender == GENDER]
            images = []

            y_pred = []
            y_true = []
            us = []

            # paths = glob(path + RACE + "\\" + GENDER + "\\*")
            paths = list(df_n.file)
            paths = [os.path.join(
                r"F:\Lab\datasets\fairface\fairface-img-margin025-trainval", x) for x in paths]

            length = len(paths)
            for i, img in enumerate(tqdm(paths)):
                in_file = open(img, 'rb')
                images.append(cv2.cvtColor(jpeg.decode(
                    in_file.read()), cv2.COLOR_BGR2RGB))
                label = int(GENDER == "Male")
                y_true.append(label)
                overall_y_true.append(label)
                in_file.close()

                if len(images) == batch_size or i == length - 1:
                    images = torch.stack([transform(image)
                                         for image in images])
                    images = images.to(device)
                    with torch.no_grad():
                        outputs = model(images)
                        feats = outputs[0]
                        features[GENDER].extend(feats)
                        outputs = outputs[1]
                        if "edl" in v:
                            alpha, outputs = torch.split(outputs, 2, dim=-1)
                            uncertainty = alpha.shape[1] / \
                                torch.sum(alpha, 1, keepdim=True)
                            us.extend(uncertainty)

                        y_pred.extend(outputs)
                        overall_y_pred.extend(outputs)
                    images = []

        # a = torch.stack(features["Male"])
        # b = torch.stack(features["Female"])
        # a = a.repeat(len(features["Female"]), 1).to("cuda:1")
        # b = b.repeat_interleave(len(features["Male"]), dim=0).to("cuda:1")

        # print(a.shape, b.shape)
        # temp_jsdivs = 0
        # for i in range(0, a.shape[0], 600000):
        #     start = i
        #     end = i + 600000 if a.shape[0] > i+600000 else a.shape[0]
        #     print(a[start:end, :].shape, b[start:end, :].shape)

        #     temp_jsdivs += jsdiv(a[start:end, :], b[start:end, :]).mean() * (end - start) / a.shape[0]
        # del a, b
        # torch.cuda.empty_cache()

        temp_jsdivs = []
        for m_feat in tqdm(features["Male"]):
            for f_feat in features["Female"]:
                temp_jsdivs.append(jsdiv(m_feat, f_feat))
        temp_jsdivs = torch.stack(temp_jsdivs)
        jsdivs.append(temp_jsdivs.mean())
        jsdivs_dict[RACE] = temp_jsdivs.mean()

    print(jsdivs_dict)


if __name__ == "__main__":
    main() 


# {'East Asian': tensor(5.5705e-07, device='cuda:0'), 'White': tensor(5.5511e-07, device='cuda:0'), 'Latino_Hispanic': tensor(5.6058e-07, device='cuda:0'), 'Southeast Asian': tensor(5.5674e-07, device='cuda:0'), 'Black': tensor(5.2634e-07, device='cuda:0'), 'Indian': tensor(5.6125e-07, device='cuda:0'), 'Middle Eastern': tensor(5.7955e-07, device='cuda:0')}
