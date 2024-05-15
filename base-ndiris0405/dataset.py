from re import sub
import tensorflow as tf
import config
from utils import randaugment
import os
import pandas as pd
import glob
import sklearn
import json
import tensorflow_io as tfio

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
                elem2 = gender_df[gender_df['subject'] == elem1['subject'].iloc[0]].sample()
                while elem1['file'].iloc[0] == elem2['file'].iloc[0]:
                    elem2 = gender_df[gender_df['subject'] == elem1['subject'].iloc[0]].sample()
                res.append((elem1['file'].iloc[0], elem2['file'].iloc[0], 1))
            

            for _ in range(num):
                elem1 = gender_df.sample()
                elem2 = gender_df[gender_df['subject'] != elem1['subject'].iloc[0]].sample()
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
            
            
            
            if split == 'train':
                df = pd.read_csv(self.labels + 'train.csv')
            else:
                df = pd.read_csv(self.labels + 'test.csv')
            
            df['gender'] = df['gender'].apply(lambda x: int(x == "M"))

            ratio = config.config["ratio"]
            if not test:
                if ratio == "F25M75":
                    male_df = df[df['gender']==1]
                    female_df = df[df['gender']==0]
                    end_ind = len(male_df) // 3
                    df = pd.concat([male_df , female_df.iloc[:end_ind]])
                elif ratio == "F75M25":
                    male_df = df[df['gender']==1]
                    female_df = df[df['gender']==0]
                    end_ind = len(female_df) // 3
                    df = pd.concat([male_df.iloc[:end_ind] , female_df])
                elif ratio == 'F100':
                    male_df = df[df['gender']==1]
                    female_df = df[df['gender']==0]
                    end_ind = len(female_df) // 3
                    df = female_df
                elif ratio == 'M100':
                    male_df =  df[df['gender']==1]
                    female_df = df[df['gender']==0]
                    end_ind = len(female_df) // 3
                    df = male_df
            if test:
                test_mode = config.config["test_mode"]
                if test_mode == 1:
                    df = df[df['gender'] == 1]
                elif test_mode == 2:
                    df = df[df['gender'] == 0]

            print("Male : {}, Female: {}".format(len(df[df['gender'] ==1]), len(df[df['gender']==0])))

            class_type = config.config["class"]
            if class_type == "gender":
                y = df['gender']
            elif class_type == "subject":
                y = df['subject']
                            
            x = df['file']
            # x = [os.path.join(self.train_dir, X) for X in x]

            
            list_ds = tf.data.Dataset.from_tensor_slices(x)
            image_count = (len(list_ds))

        
            list_ds = tf.data.Dataset.zip(
                (list_ds, tf.data.Dataset.from_tensor_slices(y)))
            list_ds = list_ds.shuffle(
                image_count, reshuffle_each_iteration=False)
            return list_ds

        fold = config.config["fold"]
        self.train_ds = get_data('train')
        image_count = (len(self.train_ds))
        class_type = config.config["class"]
        if class_type == "subject":
            val_size = int(image_count * 0.2)
            list_ds = self.train_ds
            self.train_ds = list_ds.skip(val_size)
            self.val_ds = list_ds.take(val_size)
        else:
            self.val_ds = get_data('val')
        self.test_ds = get_data('test', True)

        if verbose:
            for f in self.train_ds.take(5):
                print(f[0].numpy(), f[1].numpy())


        self.train_ds = self.train_ds.map(
            lambda x, y: self.process_path((x, y), True), num_parallel_calls=self.AUTOTUNE)
        self.val_ds = self.val_ds.map(lambda x, y: self.process_path(
            (x, y), False), num_parallel_calls=self.AUTOTUNE)
        self.test_ds = self.test_ds.map(lambda x, y: self.process_path(
            (x, y), False), num_parallel_calls=self.AUTOTUNE)

        if verbose:
            for image, label in self.train_ds.take(1):
                print("Image shape: ", image.numpy().shape)
                print("Label: ", label.numpy())
        


        if verbose:
            print("Train Size : {}".format(
                tf.data.experimental.cardinality(self.train_ds).numpy()))
            print("Validation Size : {}".format(
                tf.data.experimental.cardinality(self.val_ds).numpy()))
            print("Test Size : {}".format(
                tf.data.experimental.cardinality(self.test_ds).numpy()))

        self.train_ds = self.configure_for_performance(self.train_ds)
        self.val_ds = self.configure_for_performance(self.val_ds)
        self.test_ds = self.configure_for_performance(self.test_ds)
        if class_type != 'subject':
            self.val_ds = self.test_ds

    def load_unab(self):
        df = pd.read_csv(r"F:\Lab\nfs\base-ndiris0405\unab_left.csv", header=None)

        print(df.head())
        test_mode = config.config["test_mode"]
        if test_mode == 1:
            df = df[df[1] == 1]
        elif test_mode == 2:
            df = df[df[1] == 0]

        x = df[0]
        y = df[1]

        print("Male : {}, Female: {}".format(len(df[df[1] ==1]), len(df[df[1]==0])))
    

        list_ds = tf.data.Dataset.from_tensor_slices(x)
    
        self.test_ds = tf.data.Dataset.zip(
            (list_ds, tf.data.Dataset.from_tensor_slices(y)))

        for data in self.test_ds.take(1):
            print(data)
        self.test_ds = self.test_ds.map(lambda x, y: self.process_path(
            (x, y), False, True), num_parallel_calls=self.AUTOTUNE)

        self.test_ds = self.configure_for_performance(self.test_ds)

    def get_label(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        one_hot = parts[-2] == self.class_names
        return tf.argmax(one_hot)

    def decode_img(self, img, training, unab=False):
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
        if unab:
            img = tf.image.decode_bmp(img, channels=3)
        else:
            img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, (self.img_height, self.img_width), method='nearest')

        if training and self.randaugment:
            img = randaugment.distort_image_with_randaugment(
                img, self.randaugment_layers, self.randaugment_magnitude)
        
        # Preprocess image here
        img = tf.cast(img, tf.float32)
        img = preprocess_dict[model_type](
            img
        )
        return img

    def process_path(self, data, training=True, unab=False):
        file_path, label = data
        # label = self.get_label(file_path)
        img = tf.io.read_file(file_path)
        img = self.decode_img(img, training, unab)
        
        return img, label

    def process_verification(self, file_path):
        img = tf.io.read_file(file_path)
        img = self.decode_img(img, False)
        return img

    def configure_for_performance(self, ds, training=True):
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        ds = ds.with_options(options)
        ds = ds.cache()
        if training:
            ds = ds.shuffle(buffer_size=self.buffer_size)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)
        return ds
