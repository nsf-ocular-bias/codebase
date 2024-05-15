import numpy as np
import dataset
from callbacks import *
from datetime import datetime
import os
import config
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4, DenseNet121, MobileNetV2, InceptionV3, InceptionResNetV2, ResNet50, VGG19, Xception, EfficientNetB0
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from pyeer.eer_info import get_eer_stats
from pyeer.report import generate_eer_report
from pyeer.plot import plot_eer_stats
from tqdm import tqdm
import pandas as pd
import sklearn

# Conv2DTranspose then DownSample then again Conv2DTranspose, examine the effect


class Module(object):
    def __init__(self):
        self.img_width = config.config["input"]["image_width"]
        self.img_height = config.config["input"]["image_height"]
        self.channels = config.config["input"]["channels"]

        self.strategy = tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
        print('Number of devices: {}'.format(
            self.strategy.num_replicas_in_sync))

        self.ckpt_dir = config.config["train"]["ckpt_dir"]
        if not os.path.exists(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)

        self.run_id = config.config["run_id"]

        self.ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.ckpt_dir + "\\" + self.run_id + ".{epoch:02d}-{val_loss:.4f}.hdf5", save_freq="epoch",
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss",
        )

        self.logdir = "logs\\" + self.run_id + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = LRTensorBoard(log_dir=self.logdir, profile_batch=0)

        lr_schedule = config.config["optimizer"]["lr_scheduler"]
        initial_learning_rate = config.config["optimizer"]["initial_learning_rate"]
        first_decay_steps = config.config["optimizer"]["first_decay_steps"]
        decay_rate = config.config["optimizer"]["decay_rate"]

        if lr_schedule == "cosine_decay_warm_restart":
            self.lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate, first_decay_steps)
        elif lr_schedule == "cosine_decay":
            self.lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate, first_decay_steps)
        elif lr_schedule == "exponential_decay":
            self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate, first_decay_steps, decay_rate)

        else:
            self.lr_schedule = initial_learning_rate

        self.early_stopping = tf.keras.callbacks.EarlyStopping(patience=20)
        self.compile_model()

    class CustomModel(tf.keras.Model):
        def train_step(self, data):
            return super().train_step(data)

        def test_step(self, data):
            return super().test_step(data)

    def create_model(self):
        model_type = config.config["model"]
        model_dict = {
            "EfficientNetB4": EfficientNetB4,
            "DenseNet121": DenseNet121,
            "MobileNetV2": MobileNetV2,
            "InceptionV3": InceptionV3, 
            "InceptionResNetV2": InceptionResNetV2, 
            "ResNet50": ResNet50, 
            "VGG19": VGG19, 
            "Xception": Xception, 
            "EfficientNetB0": EfficientNetB0
        }
        class_type = config.config["class"]
        fc_classes, fc_activation = (
            1, 'sigmoid') if class_type == 'gender' else (521, 'softmax')
        base = model_dict[model_type](weights='imagenet', include_top=False,
                        input_shape=(self.img_height, self.img_width, self.channels))
        base.trainable = False
        self.model = tf.keras.models.Sequential(
            [
                base,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(1024),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.Dense(fc_classes, activation=fc_activation)
            ]
        )

    def compile_model(self):
        class_type = config.config["class"]
        loss = 'binary_crossentropy' if class_type == 'gender' else 'sparse_categorical_crossentropy'
        with self.strategy.scope():
            self.create_model()
            self.model.compile(
                loss=loss,
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=self.lr_schedule),
                metrics=["accuracy"],
            )
        self.model.summary()

    def unfreeze_model(self):
        class_type = config.config["class"]
        loss = 'binary_crossentropy' if class_type == 'gender' else 'sparse_categorical_crossentropy'
        with self.strategy.scope():
            for layer in self.model.layers[-10:]:
                if not isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = True
            self.model.compile(
                loss=loss,
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=self.lr_schedule * 1e-1),
                metrics=["accuracy"],
            )
        self.model.summary()

    def test_verification(self):
        dataloader = dataset.DataLoader()
        suba_ds, subb_ds, labels = dataloader.load_verification()
        test_model = tf.keras.Model(self.model.inputs, self.model.layers[-2].output)
        test_model.summary()

        # print(len(labels), labels)
        zipped = tf.data.Dataset.zip((suba_ds, subb_ds))
        cosine_loss = tf.keras.losses.CosineSimilarity(axis=1, reduction=tf.keras.losses.Reduction.NONE)
        scores = []
        for a_ds, b_ds in tqdm(zipped):
            a = test_model.predict(a_ds)
            b = test_model.predict(b_ds)
            # score = sklearn.metrics.pairwise.cosine_similarity(a, b)
            score = cosine_loss(a, b).numpy()
            scores.extend(list(score))
        scores = pd.Series(scores)
        genuine_scores = scores[labels == 0]
        impostor_scores = scores[labels == 1]
        # print(genuine_scores, impostor_scores)
        print(len(genuine_scores), len(impostor_scores), len(scores))

        eer_stats = get_eer_stats(genuine_scores, impostor_scores)
        plot_eer_stats([eer_stats], [self.run_id])
        generate_eer_report([eer_stats], [self.run_id], self.run_id + ".csv")

    def calc_race(self):
        dataloader = dataset.DataLoader()
        y_true = []
        y_pred = []
        r_true = []
        for x, y, r in tqdm(dataloader.test_ds):
            y_true.extend(y)
            y_pred.extend(self.model.predict(x))
            r_true.extend(r)
        
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        y_pred = np.where(y_pred > 0.5, 1,0)

        correct = (y_true == y_pred)
        race_data = pd.read_excel(r"F:\Lab\datasets\visob\demographic information_VISOB.xlsx")
        race_dict  = {}
        races = race_data.Ethnicity.unique()
        print(races)
        for race in range(len(races)):
            correct_race = correct[r_true == race]
            acc = correct_race.sum()*100/len(correct_race)
            print(acc, race)

    def test_model(self):
        self.calc_race()
        return
        if config.config["class"] == "subject":
            self.test_verification()
            return
        dataloader = dataset.DataLoader()
        self.history = self.model.evaluate(dataloader.test_ds)
        y_true = []
        y_pred = []
        for x, y in tqdm(dataloader.test_ds):
            y_true.extend(y)
            y_pred.extend(self.model.predict(x))
        threshold = 0.5
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        for i in np.unique(y_true):
            print(np.count_nonzero(y_true == i), end=' ')

        auc = roc_auc_score(y_true, y_pred)
        g_scores = y_pred[y_true == 1]
        i_scores = y_pred[y_true == 0]
        eer_stats = get_eer_stats(g_scores, i_scores)
        print(eer_stats)
        generate_eer_report([eer_stats], [self.run_id], self.run_id + ".csv")
        y_pred = np.where(y_pred > threshold, 1,0)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        print("ROC AUC : {}, F1 : {}, Precision : {}, Recall: {}, Accuracy : {}".format(auc, f1, precision, recall, acc))
        
        df = pd.read_csv(self.run_id + ".csv")

        df["Accuracy"] = acc
        df["Precision"] = precision
        df["F1"] = f1
        df["Recall"] = recall
        df["ROC AUC"] = auc

        df.to_csv(self.run_id + ".csv")

    def train_model(self, epochs=30, unfreeze_epoch=20, steps_per_epoch=None, initial_epoch=0):
        finetune = config.config["fine-tune"]
        batch_size = config.config['train']['batch_size']
        if finetune:
            weight = config.config['train']['pretrained_weight']
            self.load_pretrained(weight)
            batch_size = batch_size // 4
            self.unfreeze_model()

        dataloader = dataset.DataLoader(batch_size=batch_size)

        self.history = self.model.fit(
            dataloader.train_ds,
            validation_data=dataloader.val_ds,
            initial_epoch=initial_epoch,
            epochs=unfreeze_epoch,
            steps_per_epoch=steps_per_epoch,
            verbose=2,
            callbacks=[self.ckpt_callback,
                       self.tensorboard_callback, self.early_stopping]
        )

    def load_pretrained(self, weight):
        print("Loading weights : " + weight)
        self.model.load_weights(weight)
