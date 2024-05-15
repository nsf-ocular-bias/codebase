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
import random
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff
import tf_explain.core as tfe
from utils import viz
import gc
from tensorflow.keras import mixed_precision

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
            filepath=self.ckpt_dir + "\\" + self.run_id + ".hdf5", save_freq="epoch",
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
            1, 'sigmoid') if class_type == 'gender' else (584, 'softmax')
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


    def test_model(self):
        test_mode = config.config['test_mode']

        if config.config["class"] == "subject":
            self.test_verification()
            return
        dataloader = dataset.DataLoader()
        self.history = self.model.evaluate(dataloader.test_ds)
        
        if test_mode != 0:
            _, acc = self.history
            df = pd.DataFrame()
            df['Experiment ID'] = [self.run_id]
            df['Accuracy'] = [acc]
            df.to_csv(self.run_id + '_' + str(test_mode) + ".csv")
            return

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
        generate_eer_report([eer_stats], [self.run_id], self.run_id + '_' + str(test_mode) + ".csv")
        y_pred = np.where(y_pred > threshold, 1,0)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        print("ROC AUC : {}, F1 : {}, Precision : {}, Recall: {}, Accuracy : {}".format(auc, f1, precision, recall, acc))
        

        df = pd.read_csv(self.run_id + '_' + str(test_mode) +  ".csv")

        df["Accuracy"] = acc
        df["Precision"] = precision
        df["F1"] = f1
        df["Recall"] = recall
        df["ROC AUC"] = auc

        df.to_csv(self.run_id + '_' + str(test_mode) + ".csv")

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

    def plot_cm(self, cm):
        labels = ["Female", "Male"]
        font_colors = ['black', 'white']
        fig = ff.create_annotated_heatmap(cm, labels, labels, showscale=True, colorscale='Viridis', font_colors=font_colors, reversescale=True)
        fig.update_layout(
            autosize=False,
            width=500,
            height=500,)
        test_mode = config.config['test_mode']
        run_id = self.run_id + '_' + str(test_mode) + '_cm'
        with open(run_id + '.json', 'w') as f:
            f.write(fig.to_json())

    def model_modifier(self, unfold):
        # Split last layer into linear and activation
        self.model.layers[-1].activation = None
        class_type = config.config["class"]

        activation = 'softmax' if class_type == 'subject' else 'sigmoid'
        self.model.add(tf.keras.layers.Activation(activation))
        
        if unfold:
            # Unify the model by removing the cascading
            base_model = self.model.layers[0]
            x = base_model.output
            for layer in self.model.layers[1:]:
                x = layer(x)
            output = x

            self.model = tf.keras.Model(base_model.input, output)
        
        loss = 'binary_crossentropy' if class_type == 'gender' else 'sparse_categorical_crossentropy'
        with self.strategy.scope():
            self.model.compile(
                loss=loss,
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=self.lr_schedule),
                metrics=["accuracy"],
            )
        self.model.summary()

    def viz_model(self):
        unfold = config.config['viz_method'] in ['grad_cam', 'guided_gradcam']
        self.model_modifier(unfold)
        max_img = 10
                
        def get_cm_data():
            if config.config['viz_method'] == 'gen_data':
                dataloader = dataset.DataLoader()
                y_true = []
                y_pred = []
                X = []
                for x, y in tqdm(dataloader.test_ds):
                    y_true.extend(y)
                    y_pred.extend(self.model.predict_on_batch(x))
                    x_ = [i.numpy() for i in x]
                    X.extend(x_)
                    tf.keras.backend.clear_session()
                    gc.collect()
                y_true = np.array(y_true).flatten()
                y_pred = np.array(y_pred).flatten()
                y_pred = (y_pred > 0.5).astype(np.int8)

                # First viz confusion matrix
                cm = confusion_matrix(y_true, y_pred)
                self.plot_cm(cm)

                tp = []
                fp = []
                fn = []
                tn = []
                # Categorize inputs into tp, fp, tn, fn
                for i in range(len(y_true)):
                    if y_true[i] == 1: 
                        if y_pred[i] == 1: # true positive
                            tp.append(X[i])
                        else: # false negative
                            fn.append(X[i])
                    else: 
                        if y_pred[i] == 1: # false positive
                            fp.append(X[i])
                        else: # true negative
                            tn.append(X[i])
                # Sanity check
                print('TP {} FP {} TN {} FN {}'.format(len(tp), len(fp), len(tn), len(fn)))
                tp = random.sample(tp, min(max_img, len(tp)))
                fp = random.sample(fp, min(max_img, len(fp)))
                fn = random.sample(fn, min(max_img, len(fn)))
                tn = random.sample(tn, min(max_img, len(tn)))
                with open(self.run_id + '.npy', 'wb') as f:
                    np.save(f, tp)
                    np.save(f, fp)
                    np.save(f, tn)
                    np.save(f, fn)
            else:
                with open(self.run_id + '.npy', 'rb') as f:
                    tp = np.load(f)
                    fp = np.load(f)
                    tn = np.load(f)
                    fn = np.load(f)

            return tp, fp, tn, fn

        imgs_tp, imgs_fp, imgs_tn, imgs_fn = get_cm_data()
        

        # Viz tf-explain methods
        tf_explain_methods = [
                                tfe.grad_cam.GradCAM(),] # Rewrite GradCAM

        # Vanilla Gradients    
        explainer = viz.VanillaGradients()
        
        viz_method = config.config['viz_method']

        if viz_method == 'vanilla_gradients':
            for i in range(max_img):
                grid = explainer.explain(([imgs_tp[i]], None), self.model, class_index=0, cls='pos') 
                explainer.save(grid, 'outputs', self.run_id + '_vanilla_gradients_pos_tp_{}.png'.format(i))
                grid = explainer.explain(([imgs_tp[i]], None), self.model, class_index=0, cls='neg') 
                explainer.save(grid, 'outputs', self.run_id + '_vanilla_gradients_neg_tp_{}.png'.format(i))

            for i in range(max_img):
                grid = explainer.explain(([imgs_fp[i]], None), self.model, class_index=0, cls='pos') 
                explainer.save(grid, 'outputs', self.run_id + '_vanilla_gradients_pos_fp_{}.png'.format(i))
                grid = explainer.explain(([imgs_fp[i]], None), self.model, class_index=0, cls='neg') 
                explainer.save(grid, 'outputs', self.run_id + '_vanilla_gradients_neg_fp_{}.png'.format(i))

            for i in range(max_img):
                try:
                    grid = explainer.explain(([imgs_fn[i]], None), self.model, class_index=0, cls='pos') 
                    explainer.save(grid, 'outputs', self.run_id + '_vanilla_gradients_pos_fn_{}.png'.format(i))
                    grid = explainer.explain(([imgs_fn[i]], None), self.model, class_index=0, cls='neg') 
                    explainer.save(grid, 'outputs', self.run_id + '_vanilla_gradients_neg_fn_{}.png'.format(i))
                except Exception as e:
                    print(e)
            for i in range(max_img):
                grid = explainer.explain(([imgs_tn[i]], None), self.model, class_index=0, cls='pos') 
                explainer.save(grid, 'outputs', self.run_id + '_vanilla_gradients_pos_tn_{}.png'.format(i))
                grid = explainer.explain(([imgs_tn[i]], None), self.model, class_index=0, cls='neg') 
                explainer.save(grid, 'outputs', self.run_id + '_vanilla_gradients_neg_tn_{}.png'.format(i))

            tf.keras.backend.clear_session()
            gc.collect()

        elif viz_method == 'occlusion_sensitivity':
            # Occlusion Sensitivity
            explainer = viz.OcclusionSensitivity()
            
            for i in range(max_img):
                grid = explainer.explain(([imgs_tp[i]], None), self.model, class_index=0, patch_size=45, cls='pos') 
                explainer.save(grid, 'outputs', self.run_id + '_occlusion_sensitivity_pos_tp_{}.png'.format(i))
                grid = explainer.explain(([imgs_tp[i]], None), self.model, class_index=0, patch_size=45, cls='neg') 
                explainer.save(grid, 'outputs', self.run_id + '_occlusion_sensitivity_neg_tp_{}.png'.format(i))

            for i in range(max_img):
                grid = explainer.explain(([imgs_fp[i]], None), self.model, class_index=0, patch_size=45, cls='pos') 
                explainer.save(grid, 'outputs', self.run_id + '_occlusion_sensitivity_pos_fp_{}.png'.format(i))
                grid = explainer.explain(([imgs_fp[i]], None), self.model, class_index=0, patch_size=45, cls='neg') 
                explainer.save(grid, 'outputs', self.run_id + '_occlusion_sensitivity_neg_fp_{}.png'.format(i))

            for i in range(max_img):
                try:
                    grid = explainer.explain(([imgs_fn[i]], None), self.model, class_index=0, patch_size=45, cls='pos') 
                    explainer.save(grid, 'outputs', self.run_id + '_occlusion_sensitivity_pos_fn_{}.png'.format(i))
                    grid = explainer.explain(([imgs_fn[i]], None), self.model, class_index=0, patch_size=45, cls='neg') 
                    explainer.save(grid, 'outputs', self.run_id + '_occlusion_sensitivity_neg_fn_{}.png'.format(i))
                except Exception as e:
                    print(e)
            for i in range(max_img):
                grid = explainer.explain(([imgs_tn[i]], None), self.model, class_index=0, patch_size=45, cls='pos') 
                explainer.save(grid, 'outputs', self.run_id + '_occlusion_sensitivity_pos_tn_{}.png'.format(i))
                grid = explainer.explain(([imgs_tn[i]], None), self.model, class_index=0, patch_size=45, cls='neg') 
                explainer.save(grid, 'outputs', self.run_id + '_occlusion_sensitivity_neg_tn_{}.png'.format(i))

            tf.keras.backend.clear_session()
            gc.collect()

        # SmoothGrad 
        elif viz_method == 'smooth_grad':
            explainer = viz.SmoothGrad()
            
            for i in range(max_img):
                grid = explainer.explain(([imgs_tp[i]], None), self.model, class_index=0, cls='pos') 
                explainer.save(grid, 'outputs', self.run_id + '_smoothgrad_pos_tp_{}.png'.format(i))
                grid = explainer.explain(([imgs_tp[i]], None), self.model, class_index=0, cls='neg') 
                explainer.save(grid, 'outputs', self.run_id + '_smoothgrad_neg_tp_{}.png'.format(i))

            for i in range(max_img):
                grid = explainer.explain(([imgs_fp[i]], None), self.model, class_index=0, cls='pos') 
                explainer.save(grid, 'outputs', self.run_id + '_smoothgrad_pos_fp_{}.png'.format(i))
                grid = explainer.explain(([imgs_fp[i]], None), self.model, class_index=0, cls='neg') 
                explainer.save(grid, 'outputs', self.run_id + '_smoothgrad_neg_fp_{}.png'.format(i))

            for i in range(max_img):
                try:
                    grid = explainer.explain(([imgs_fn[i]], None), self.model, class_index=0, cls='pos') 
                    explainer.save(grid, 'outputs', self.run_id + '_smoothgrad_pos_fn_{}.png'.format(i))
                    grid = explainer.explain(([imgs_fn[i]], None), self.model, class_index=0, cls='neg') 
                    explainer.save(grid, 'outputs', self.run_id + '_smoothgrad_neg_fn_{}.png'.format(i))
                except Exception as e:
                    print(e)
            for i in range(max_img):
                grid = explainer.explain(([imgs_tn[i]], None), self.model, class_index=0, cls='pos') 
                explainer.save(grid, 'outputs', self.run_id + '_smoothgrad_pos_tn_{}.png'.format(i))
                grid = explainer.explain(([imgs_tn[i]], None), self.model, class_index=0, cls='neg') 
                explainer.save(grid, 'outputs', self.run_id + '_smoothgrad_neg_tn_{}.png'.format(i))

            tf.keras.backend.clear_session()
            gc.collect()
        elif viz_method == 'integrated_gradients':
            # Integrated Gradients    
            explainer = viz.IntegratedGradients()
            
            for i in range(max_img):
                grid = explainer.explain(([imgs_tp[i]], None), self.model, class_index=0, cls='pos') 
                explainer.save(grid, 'outputs', self.run_id + '_integrated_gradients_pos_tp_{}.png'.format(i))
                grid = explainer.explain(([imgs_tp[i]], None), self.model, class_index=0, cls='neg') 
                explainer.save(grid, 'outputs', self.run_id + '_integrated_gradients_neg_tp_{}.png'.format(i))

            for i in range(max_img):
                grid = explainer.explain(([imgs_fp[i]], None), self.model, class_index=0, cls='pos') 
                explainer.save(grid, 'outputs', self.run_id + '_integrated_gradients_pos_fp_{}.png'.format(i))
                grid = explainer.explain(([imgs_fp[i]], None), self.model, class_index=0, cls='neg') 
                explainer.save(grid, 'outputs', self.run_id + '_integrated_gradients_neg_fp_{}.png'.format(i))

            for i in range(max_img):
                try:
                    grid = explainer.explain(([imgs_fn[i]], None), self.model, class_index=0, cls='pos') 
                    explainer.save(grid, 'outputs', self.run_id + '_integrated_gradients_pos_fn_{}.png'.format(i))
                    grid = explainer.explain(([imgs_fn[i]], None), self.model, class_index=0, cls='neg') 
                    explainer.save(grid, 'outputs', self.run_id + '_integrated_gradients_neg_fn_{}.png'.format(i))
                except Exception as e:
                    print(e)
            for i in range(max_img):
                grid = explainer.explain(([imgs_tn[i]], None), self.model, class_index=0, cls='pos') 
                explainer.save(grid, 'outputs', self.run_id + '_integrated_gradients_pos_tn_{}.png'.format(i))
                grid = explainer.explain(([imgs_tn[i]], None), self.model, class_index=0, cls='neg') 
                explainer.save(grid, 'outputs', self.run_id + '_integrated_gradients_neg_tn_{}.png'.format(i))

        elif viz_method == 'gradients_input':
            # Gradients Input
            explainer = viz.GradientsInputs()
            
            for i in range(max_img):
                grid = explainer.explain(([imgs_tp[i]], None), self.model, class_index=0, cls='pos') 
                explainer.save(grid, 'outputs', self.run_id + '_gradients_inputs_pos_tp_{}.png'.format(i))
                grid = explainer.explain(([imgs_tp[i]], None), self.model, class_index=0, cls='neg') 
                explainer.save(grid, 'outputs', self.run_id + '_gradients_inputs_neg_tp_{}.png'.format(i))
                

            for i in range(max_img):
                grid = explainer.explain(([imgs_fp[i]], None), self.model, class_index=0, cls='pos') 
                explainer.save(grid, 'outputs', self.run_id + '_gradients_inputs_pos_fp_{}.png'.format(i))
                grid = explainer.explain(([imgs_fp[i]], None), self.model, class_index=0, cls='neg') 
                explainer.save(grid, 'outputs', self.run_id + '_gradients_inputs_neg_fp_{}.png'.format(i))

            for i in range(max_img):
                try:
                    grid = explainer.explain(([imgs_fn[i]], None), self.model, class_index=0, cls='pos') 
                    explainer.save(grid, 'outputs', self.run_id + '_gradients_inputs_pos_fn_{}.png'.format(i))
                    grid = explainer.explain(([imgs_fn[i]], None), self.model, class_index=0, cls='neg') 
                    explainer.save(grid, 'outputs', self.run_id + '_gradients_inputs_neg_fn_{}.png'.format(i))
                except Exception as e:
                    print(e)
            for i in range(max_img):
                grid = explainer.explain(([imgs_tn[i]], None), self.model, class_index=0, cls='pos') 
                explainer.save(grid, 'outputs', self.run_id + '_gradients_inputs_pos_tn_{}.png'.format(i))
                grid = explainer.explain(([imgs_tn[i]], None), self.model, class_index=0, cls='neg') 
                explainer.save(grid, 'outputs', self.run_id + '_gradients_inputs_neg_tn_{}.png'.format(i))

        elif viz_method == 'grad_cam':
            # GradCAM
            explainer = viz.GradCAM()
            
            for i in range(max_img):
                try:
                    grid = explainer.explain(([imgs_tp[i]], None), self.model, class_index=0, cls='pos', use_guided_grads=False) 
                    explainer.save(grid, 'outputs', self.run_id + '_grad_cam_pos_tp_{}.png'.format(i))
                    grid = explainer.explain(([imgs_tp[i]], None), self.model, class_index=0, cls='neg', use_guided_grads=False) 
                    explainer.save(grid, 'outputs', self.run_id + '_grad_cam_neg_tp_{}.png'.format(i))
                except Exception as e:
                    print(e)

            for i in range(max_img):
                try:
                    grid = explainer.explain(([imgs_fp[i]], None), self.model, class_index=0, cls='pos', use_guided_grads=False) 
                    explainer.save(grid, 'outputs', self.run_id + '_grad_cam_pos_fp_{}.png'.format(i))
                    grid = explainer.explain(([imgs_fp[i]], None), self.model, class_index=0, cls='neg', use_guided_grads=False) 
                    explainer.save(grid, 'outputs', self.run_id + '_grad_cam_neg_fp_{}.png'.format(i))
                except Exception as e:
                    print(e)

            for i in range(max_img):
                try:
                    grid = explainer.explain(([imgs_fn[i]], None), self.model, class_index=0, cls='pos', use_guided_grads=False) 
                    explainer.save(grid, 'outputs', self.run_id + '_grad_cam_pos_fn_{}.png'.format(i))
                    grid = explainer.explain(([imgs_fn[i]], None), self.model, class_index=0, cls='neg', use_guided_grads=False) 
                    explainer.save(grid, 'outputs', self.run_id + '_grad_cam_neg_fn_{}.png'.format(i))
                except Exception as e:
                    print(e)
            for i in range(max_img):
                try:
                    grid = explainer.explain(([imgs_tn[i]], None), self.model, class_index=0, cls='pos', use_guided_grads=False) 
                    explainer.save(grid, 'outputs', self.run_id + '_grad_cam_pos_tn_{}.png'.format(i))
                    grid = explainer.explain(([imgs_tn[i]], None), self.model, class_index=0, cls='neg', use_guided_grads=False) 
                    explainer.save(grid, 'outputs', self.run_id + '_grad_cam_neg_tn_{}.png'.format(i))
                except Exception as e:
                    print(e)
        elif viz_method == 'guided_gradcam':
            # Guided GradCAM
            explainer = viz.GradCAM()
            
            for i in range(max_img):
                grid = explainer.explain(([imgs_tp[i]], None), self.model, class_index=0, cls='pos', use_guided_grads=True) 
                explainer.save(grid, 'outputs', self.run_id + '_guided_gradcam_pos_tp_{}.png'.format(i))
                grid = explainer.explain(([imgs_tp[i]], None), self.model, class_index=0, cls='neg', use_guided_grads=True) 
                explainer.save(grid, 'outputs', self.run_id + '_guided_gradcam_neg_tp_{}.png'.format(i))
                

            for i in range(max_img):
                grid = explainer.explain(([imgs_fp[i]], None), self.model, class_index=0, cls='pos', use_guided_grads=True) 
                explainer.save(grid, 'outputs', self.run_id + '_guided_gradcam_pos_fp_{}.png'.format(i))
                grid = explainer.explain(([imgs_fp[i]], None), self.model, class_index=0, cls='neg', use_guided_grads=True) 
                explainer.save(grid, 'outputs', self.run_id + '_guided_gradcam_neg_fp_{}.png'.format(i))

            for i in range(max_img):
                try:
                    grid = explainer.explain(([imgs_fn[i]], None), self.model, class_index=0, cls='pos', use_guided_grads=True) 
                    explainer.save(grid, 'outputs', self.run_id + '_guided_gradcam_pos_fn_{}.png'.format(i))
                    grid = explainer.explain(([imgs_fn[i]], None), self.model, class_index=0, cls='neg', use_guided_grads=True) 
                    explainer.save(grid, 'outputs', self.run_id + '_guided_gradcam_neg_fn_{}.png'.format(i))
                except Exception as e:
                    print(e)
            for i in range(max_img):
                grid = explainer.explain(([imgs_tn[i]], None), self.model, class_index=0, cls='pos', use_guided_grads=True) 
                explainer.save(grid, 'outputs', self.run_id + '_guided_gradcam_pos_tn_{}.png'.format(i))
                grid = explainer.explain(([imgs_tn[i]], None), self.model, class_index=0, cls='neg', use_guided_grads=True) 
                explainer.save(grid, 'outputs', self.run_id + '_guided_gradcam_neg_tn_{}.png'.format(i))
