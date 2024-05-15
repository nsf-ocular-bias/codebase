import tensorflow as tf
import config
from utils import randaugment
import os
import pandas as pd
import glob
import sklearn
import tensorflow_model_remediation.min_diff as md
import copy


class DataLoader(object):
    def __init__(self, verbose=True, batch_size=None):
        self.train_dir = config.config["input"]["train_dir"]
        # self.train_dir = os.path.join(self.train_dir, dir_)

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
        d = r"F:\Lab\datasets\UFPR-Periocular\UFPR-Periocular\experimentalProtocol\open_world_valopen\test_pairs\fold1\*"

        from glob import glob

        # files = glob(d)
        # dfs = []
        # for f in files:
        #     df = pd.read_csv(f, header=None, sep=' ')
        #     df[0] = df[0].apply(lambda x:x.split(".")[0][:-1] + '.jpg')
        #     df[1] = df[1].apply(lambda x:x.split(".")[0][:-1] + '.jpg')

        #     df[0] = df[0].apply(lambda x: os.path.join(self.train_dir, x))
        #     df[1] = df[1].apply(lambda x: os.path.join(self.train_dir, x))
        #     df.drop(df[df[0] == df[1]].index, inplace = True)
        #     dfs.append(df)
        # df = pd.concat(dfs)

        test_mode = config.config["test_mode"]
        gender_df = pd.read_csv(
            r"F:\Lab\datasets\UFPR-Periocular\UFPR-Periocular\experimentalProtocol\open_world_valopen\test_fold1.txt", header=None, sep=' ', usecols=[0, 3])
        gender_df[0] = gender_df[0].apply(
            lambda x: x.split('.')[0][:-1] + '.jpg')
        gender_df[2] = gender_df[0].apply(lambda x: x[1:5])
        if test_mode == 1:  # Male only
            # Filter female queries out
            gender_df = gender_df[gender_df[3] == 1]

        elif test_mode == 2:
            # Filter male queries out
            gender_df = gender_df[gender_df[3] == 0]

        def generate(gender_df, num=5000):
            res = []
            from random import choice

            # select same
            for _ in range(num):
                elem1 = gender_df.sample()
                elem2 = gender_df[gender_df[2] == elem1[2].iloc[0]].sample()
                while elem1[0].iloc[0] == elem2[0].iloc[0]:
                    elem2 = gender_df[gender_df[2]
                                      == elem1[2].iloc[0]].sample()
                res.append((elem1[0].iloc[0], elem2[0].iloc[0], 1))

            for _ in range(num):
                elem1 = gender_df.sample()
                elem2 = gender_df[gender_df[2] != elem1[2].iloc[0]].sample()
                res.append((elem1[0].iloc[0], elem2[0].iloc[0], 0))
            return res
        df = generate(gender_df, 5000)
        df = pd.DataFrame(df)
        df[0] = df[0].apply(lambda x: os.path.join(self.train_dir, x))
        df[1] = df[1].apply(lambda x: os.path.join(self.train_dir, x))

        # df_true = df[df[2] == 1]
        # df_false = df[df[2] == 0]

        # df_true = df_true.sample(frac=0.052).reset_index(drop=True)
        # df_false = df_false.sample(frac=0.001).reset_index(drop=True)
        # df = pd.concat([df_true, df_false])
        # df = df.reset_index(drop=True)

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

    def load_files(self, verbose=True, race="all", gender="all"):

        # Get training data
        def get_data(split, test=False):
            if split == "train":
                df = pd.read_csv(os.path.join(
                    config.config["input"]["labels"], "fairface_label_train.csv"))
                files_path = config.config["input"]["train_dir"]
            else:
                df = pd.read_csv(os.path.join(
                    config.config["input"]["labels"], "fairface_label_val.csv"))
                files_path = config.config["input"]["eval_dir"]

            if race != "all":
                    df = df[df["race"] == race]
            if gender != "all":
                df = df[df["gender"] == gender]
            df["file"] = df["file"].apply(
                lambda x: os.path.join(files_path, x)).tolist()
            df["gender"] = df["gender"].apply(
                lambda x: 1 if x == "Male" else 0)

            if config.config["min_diff"] and split == "train":
                minority_mask = df["race"].apply(lambda x: x == "Black")
                majority_mask = df["race"].apply(lambda x: x != "Black")
                true_negative_mask = df["gender"] == 0
                data_train_main = copy.copy(df)
                data_train_sensitive = df[minority_mask & true_negative_mask]
                data_train_nonsensitive = df[majority_mask &
                                             true_negative_mask]

                data_train_main_ds = tf.data.Dataset.from_tensor_slices(
                    (data_train_main["file"], data_train_main["gender"]))
                data_train_sensitive_ds = tf.data.Dataset.from_tensor_slices(
                    (data_train_sensitive["file"], data_train_sensitive["gender"]))
                data_train_nonsensitive_ds = tf.data.Dataset.from_tensor_slices(
                    (data_train_nonsensitive["file"], data_train_nonsensitive["gender"]))

                data_train_main_ds = data_train_main_ds.map(
                    lambda x, y: self.process_path((x, y), True), num_parallel_calls=self.AUTOTUNE)
                data_train_sensitive_ds = data_train_sensitive_ds.map(
                    lambda x, y: self.process_path((x, y), True), num_parallel_calls=self.AUTOTUNE)
                data_train_nonsensitive_ds = data_train_nonsensitive_ds.map(
                    lambda x, y: self.process_path((x, y), True), num_parallel_calls=self.AUTOTUNE)

                data_train_main_ds = self.configure_for_performance(data_train_main_ds)
                data_train_sensitive_ds = self.configure_for_performance(data_train_sensitive_ds)
                data_train_nonsensitive_ds = self.configure_for_performance(data_train_nonsensitive_ds)
                list_ds = md.keras.utils.input_utils.pack_min_diff_data(
                    data_train_main_ds, data_train_sensitive_ds, data_train_nonsensitive_ds)
            else:
                
                x = df["file"]
                y = df["gender"]
                list_ds = tf.data.Dataset.from_tensor_slices(x)

                list_ds = tf.data.Dataset.zip(
                    (list_ds, tf.data.Dataset.from_tensor_slices(y)))

                image_count = (len(list_ds))

                if not test:
                    list_ds = list_ds.shuffle(
                        image_count, reshuffle_each_iteration=False)
                list_ds = list_ds.map(
                    lambda x, y: self.process_path((x, y), not test), num_parallel_calls=self.AUTOTUNE)
                list_ds = self.configure_for_performance(list_ds)

            return list_ds

        fold = config.config["fold"]

        self.train_ds = get_data(
            "train")
        self.val_ds = get_data("val", True)
        self.test_ds = get_data("val", True)

        # if verbose:
        #     for image, label in self.train_ds.take(1):
        #         print("Image shape: ", image.numpy().shape)
        #         print("Label: ", label.numpy())

        if verbose:
            print("Train Size : {}".format(
                tf.data.experimental.cardinality(self.train_ds).numpy()))
            print("Validation Size : {}".format(
                tf.data.experimental.cardinality(self.val_ds).numpy()))
            print("Test Size : {}".format(
                tf.data.experimental.cardinality(self.test_ds).numpy()))

        # self.train_ds = self.configure_for_performance(self.train_ds)
        # self.val_ds = self.configure_for_performance(self.val_ds)
        # self.test_ds = self.configure_for_performance(self.test_ds)

    


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
        img = tf.image.decode_jpeg(img, channels=3)
        if training and self.randaugment:
            img = randaugment.distort_image_with_randaugment(
                img, self.randaugment_layers, self.randaugment_magnitude)
        # img = tf.image.resize(img, (256, 128))

        # Preprocess image here
        img = tf.cast(img, tf.float32)
        img = preprocess_dict[model_type](
            img
        )
        return img

    def process_path(self, data, training=True):
        file_path, label = data
        # label = self.get_label(file_path)
        img = tf.io.read_file(file_path)
        img = self.decode_img(img, training)

        return img, label

    def process_verification(self, file_path):
        img = tf.io.read_file(file_path)
        img = self.decode_img(img, False)
        return img

    def configure_for_performance(self, ds, training=True):
        # ds = ds.cache()
        if training:
            ds = ds.shuffle(buffer_size=self.buffer_size)
        ds = ds.batch(self.batch_size)
        # ds = ds.prefetch(buffer_size=self.AUTOTUNE)
        return ds
