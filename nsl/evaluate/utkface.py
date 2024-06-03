import json

with open("evaluate\\weights.json", "r") as f:
    weights = json.load(f)

import click
import pandas as pd
import numpy as np
path = r"F:\Lab\datasets\UTKFace\\"
import os

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
    import torch
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

    model = TrainableModel.load_from_checkpoint(ckpt_path, config=config).to(device)
    model.eval()


    df = pd.read_csv(r"F:\Lab\datasets\utkface_all.csv")
    from sklearn.model_selection import train_test_split
    _, df = train_test_split(df, test_size=0.2, random_state=42)
    overall_y_pred = []
    overall_y_true = []
    accs = []
    for RACE in ["White", "Black", "Asian", "Indian", "Others"]:
        df_ = df[df.race == RACE]
        for GENDER in ["Male", "Female"]:
            df_n = df_[df_.gender == GENDER]
            images = []

            y_pred = []
            y_true = []
            us = []

            # paths = glob(path + RACE + "\\" + GENDER + "\\*")
            paths = list(df_n.file)
            paths = [os.path.join(r"F:\Lab\datasets\UTKFace", x) for x in paths]

            length = len(paths)
            for i, img in enumerate(tqdm(paths)):
                in_file = open(img, 'rb')
                images.append(cv2.cvtColor(jpeg.decode(in_file.read()), cv2.COLOR_BGR2RGB))
                label = int(GENDER == "Male")
                y_true.append(label)
                overall_y_true.append(label)
                in_file.close()

                # dets = detector(img, 1)
                # faces = dlib.full_object_detections()

                # for detection in dets:
                #     faces.append(sp(img, detection))
                


                if len(images) == batch_size or i == length - 1:
                    images = torch.stack([transform(image) for image in images])
                    images = images.to(device)
                    with torch.no_grad():
                        outputs = model(images)
                        outputs = outputs[1]
                        if "edl" in v:
                            alpha, outputs = torch.split(outputs, 2, dim=-1)
                            uncertainty =  alpha.shape[1] / torch.sum(alpha, 1, keepdim=True)
                            us.extend(uncertainty)

                        y_pred.extend(outputs)
                        overall_y_pred.extend(outputs)
                    images = []


            y_pred = torch.stack(y_pred).cpu()
            y_true = torch.tensor(y_true).cpu()
            # us = torch.stack(us).cpu().flatten()
            from torchmetrics.functional import accuracy,confusion_matrix

            acc = accuracy(y_pred.argmax(1), y_true)

            print(RACE, GENDER, acc)
            # acc = accuracy(y_pred.argmax(1)[us < 0.03], y_true[us < 0.03])

            # print(RACE, GENDER, acc, len(y_true[us < 0.03]))
            accs.append(acc)

    print(np.std(np.array(accs)) * 100) 

    y_pred = torch.stack(overall_y_pred).cpu()
    y_true = torch.tensor(overall_y_true).cpu()
    # print(y_pred.shape, y_true.shape)
    acc = accuracy(y_pred.argmax(1), y_true)
    print(acc)


if __name__ == "__main__":
    main()