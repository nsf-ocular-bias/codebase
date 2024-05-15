from re import sub
import tensorflow as tf
import config
from utils import randaugment
import os
import pandas as pd
import glob
import sklearn
import json


class DataLoader(object):
    def __init__(self, verbose=True, batch_size=None):
        self.type = config.config["input"]["type"]
        self.train_dir = config.config["input"]["train_dir"]

        self.img_height = config.config["input"]["image_height"]
        self.img_width = config.config["input"]["image_width"]
        self.AUTOTUNE = tf.data.AUTOTUNE
        self.randaugment = config.config["input"]["randaugment"]
        self.randaugment_layers = config.config["input"]["randaugment_layers"]
        self.randaugment_magnitude = config.config["input"]["randaugment_magnitude"]
        self.batch_size = batch_size if batch_size else config.config["train"]["batch_size"]
        self.buffer_size = config.config["train"]["buffer_size"]
        self.labels = config.config["input"]["labels"]

        # Set class names here
        self.class_names = ["real", "fake"]

        self.load_files(verbose)

    def load_verification(self):

        df = pd.read_csv(self.labels + 'test.csv')

        test_mode = config.config["test_mode"]
        if test_mode == 1:
            df = df[df['gender'] == 1]
        elif test_mode == 2:
            df = df[df['gender'] == 0]

        gender_df = df

        def generate(gender_df, num=5000):
            res = []
            from random import choice

            # select same
            for _ in range(num):
                elem1 = gender_df.sample()
                elem2 = gender_df[gender_df['subject']
                                  == elem1['subject'].iloc[0]].sample()
                while elem1['file'].iloc[0] == elem2['file'].iloc[0]:
                    elem2 = gender_df[gender_df['subject']
                                      == elem1['subject'].iloc[0]].sample()
                res.append((elem1['file'].iloc[0], elem2['file'].iloc[0], 1))

            for _ in range(num):
                elem1 = gender_df.sample()
                elem2 = gender_df[gender_df['subject']
                                  != elem1['subject'].iloc[0]].sample()
                res.append((elem1['file'].iloc[0], elem2['file'].iloc[0], 0))
            return res
        df = generate(gender_df, 5000)
        df = pd.DataFrame(df)
        df[0] = df[0].apply(lambda x: os.path.join(self.train_dir, x))
        df[1] = df[1].apply(lambda x: os.path.join(self.train_dir, x))

        suba_ds = tf.data.Dataset.from_tensor_slices(df[0])
        subb_ds = tf.data.Dataset.from_tensor_slices(df[1])

        suba_ds = suba_ds.map(lambda x: self.process_verification(
            x), num_parallel_calls=self.AUTOTUNE)
        subb_ds = subb_ds.map(lambda x: self.process_verification(
            x), num_parallel_calls=self.AUTOTUNE)

        suba_ds = self.configure_for_performance(suba_ds, training=False)
        subb_ds = self.configure_for_performance(subb_ds, training=False)

        labels = tf.data.Dataset.from_tensor_slices(df[2])

        return suba_ds, subb_ds, df[2]

    def load_files(self, verbose=True):

        # Get training data
        def get_data(split, test=False):
            if split == "train":
                filename = config.config["input"]["train_dir"]
            else:
                filename = config.config["input"]["eval_dir"]
            ds = tf.data.TFRecordDataset(filename)

            ds_desc = {
                'gender': tf.io.FixedLenFeature([], tf.string),
                'image': tf.io.FixedLenFeature([], tf.string),
                'subject': tf.io.FixedLenFeature([], tf.int64),
            }

            def _parse_tfrecord(example_proto):
                return tf.io.parse_single_example(example_proto, ds_desc)

            ds = ds.map(_parse_tfrecord, num_parallel_calls=self.AUTOTUNE)

            if not test:
                ratio = config.config["ratio"]
                target_dist = [0.5, 0.5]
                if ratio == "F25M75":
                    target_dist = [0.25, 0.75]
                elif ratio == "F75M25":
                    target_dist = [0.75, 0.25]
                elif ratio == 'F100':
                    target_dist = [1.0, 0.0]
                elif ratio == 'M100':
                    target_dist = [0.0, 1.0]

                initial_dist = [0.5, 0.5]

                resampler = tf.data.experimental.rejection_resample(
                    class_func=lambda x: tf.cast(x["gender"] == "M", tf.int64),
                    target_dist=target_dist,
                    initial_dist=initial_dist)

                ds = ds.apply(resampler)
                ds = ds.map(lambda _, x: x)

            # male_ds = ds.filter(lambda x: x["gender"] == "M")
            # female_ds = ds.filter(lambda x: x["gender"] == "F")

            # print("Male : ", male_ds.reduce(0, lambda x, _: x + 1))
            # print("Female : ", female_ds.reduce(0, lambda x, _: x + 1))


            if test:
                test_mode = config.config["test_mode"]
                if test_mode == 1:
                    ds = ds.filter(lambda x: x["gender"] == "M", )
                elif test_mode == 2:
                    ds = ds.filter(lambda x: x["gender"] == "F")

            return ds

        self.train_ds = get_data('train')
        image_count = self.train_ds.reduce(0, lambda x, _: x + 1)
        class_type = config.config["class"]
        if class_type == "subject":
            val_size = int(image_count * 0.2)
            list_ds = self.train_ds
            self.train_ds = list_ds.skip(val_size)
            self.val_ds = list_ds.take(val_size)
        else:
            self.val_ds = get_data('val', True)
        self.test_ds = get_data('test', True)



        self.train_ds = self.train_ds.map(
            lambda x: self.process_path(x, True), num_parallel_calls=self.AUTOTUNE)
        self.val_ds = self.val_ds.map(lambda x: self.process_path(
            x, False), num_parallel_calls=self.AUTOTUNE)
        self.test_ds = self.test_ds.map(lambda x: self.process_path(
            x, False), num_parallel_calls=self.AUTOTUNE)

        if verbose:
            for image, label in self.train_ds.take(1):
                print("Image shape: ", image.numpy().shape)
                print("Label: ", label.numpy())
        if verbose:
            for image, label in self.val_ds.take(1):
                print("Image shape: ", image.numpy().shape)
                print("Label: ", label.numpy())

        self.train_ds = self.configure_for_performance(self.train_ds)
        self.val_ds = self.configure_for_performance(self.val_ds)
        self.test_ds = self.configure_for_performance(self.test_ds)
        
        if class_type != 'subject':
            self.val_ds = self.test_ds

    def get_label(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        one_hot = parts[-2] == self.class_names
        return tf.argmax(one_hot)

    def decode_img(self, img, training):
        preprocess_dict = {
            "EfficientNetB4": tf.keras.applications.efficientnet.preprocess_input,
            "DenseNet121": tf.keras.applications.densenet.preprocess_input,
            "MobileNetV2": tf.keras.applications.mobilenet_v2.preprocess_input,
            "InceptionV3": tf.keras.applications.inception_v3.preprocess_input,
            "InceptionResNetV2": tf.keras.applications.inception_resnet_v2.preprocess_input,
            "ResNet50": tf.keras.applications.resnet50.preprocess_input,
            "VGG19": tf.keras.applications.vgg19.preprocess_input,
            "Xception": tf.keras.applications.xception.preprocess_input,
            "EfficientNetB0": tf.keras.applications.efficientnet.preprocess_input
        }
        model_type = config.config["model"]
        img = tf.io.parse_tensor(img, out_type=tf.uint8)
        img = tf.reshape(img, shape=[480,640,4])
        img = img[:,:,:3]

        if training and self.randaugment:
            img = randaugment.distort_image_with_randaugment(
                img, self.randaugment_layers, self.randaugment_magnitude)
        img = tf.image.resize(
            img, (self.img_height, self.img_width), method='nearest')

        # Preprocess image here
        img = tf.cast(img, tf.float32)
        img = preprocess_dict[model_type](
            img
        )
        return img

    def process_path(self, data, training=True):
        img = data["image"]
        if config.config["class"] == "gender":
            label = tf.cast(data["gender"] == "M", tf.int32)
        img = self.decode_img(img, training)
        return img, label

    def process_verification(self, file_path):
        img = tf.io.read_file(file_path)
        img = self.decode_img(img, False)
        return img

    def configure_for_performance(self, ds, training=True):
        # options = tf.data.Options()
        # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        # ds = ds.with_options(options)
        ds = ds.cache()
        if training:
            ds = ds.shuffle(buffer_size=self.buffer_size)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)
        return ds
