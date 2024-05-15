import tensorflow as tf
import config
from utils import randaugment
import os
import pandas as pd
import glob
import sklearn


class DataLoader(object):
    def __init__(self, verbose=True, batch_size=None):
        self.type = config.config["input"]["type"]
        self.train_dir = config.config["input"]["train_dir"]
        if self.type == "aligned":
            dir_ = "aligned"
        else:
            dir_ = "periocularCropped"
        self.train_dir = os.path.join(self.train_dir, dir_)

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
        gender_df = pd.read_csv(r"F:\Lab\datasets\UFPR-Periocular\UFPR-Periocular\experimentalProtocol\open_world_valopen\test_fold1.txt", header=None, sep=' ', usecols=[0, 3])
        gender_df[0] = gender_df[0].apply(lambda x: x.split('.')[0][:-1] + '.jpg')
        gender_df[2] = gender_df[0].apply(lambda x: x[1:5])
        if test_mode == 1: # Male only
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
                    elem2 = gender_df[gender_df[2] == elem1[2].iloc[0]].sample()
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

        


    def load_files(self, verbose=True):
        
        # Get training data
        def get_data(split, test=False):
            dfs = []
            dir_ = self.labels + split 
            print(dir_)
            for file in glob.glob(dir_):
                dfs.append(pd.read_csv(file, sep=' ', header=None))
            print(dfs)
            df = pd.concat(dfs, sort=False)
            df = sklearn.utils.shuffle(df)
            print(df.head())

            ratio = config.config["ratio"]
            if not test:
                if ratio == "F25M75":
                    male_df = df[df[3]==1]
                    female_df = df[df[3]==0]
                    end_ind = len(male_df) // 3
                    df = pd.concat([male_df , female_df.iloc[:end_ind]])
                elif ratio == "F75M25":
                    male_df = df[df[3]==1]
                    female_df = df[df[3]==0]
                    end_ind = len(female_df) // 3
                    df = pd.concat([male_df.iloc[:end_ind] , female_df])
                elif ratio == 'F100':
                    male_df = df[df[3]==1]
                    female_df = df[df[3]==0]
                    end_ind = len(female_df) // 3
                    df = female_df
                elif ratio == 'M100':
                    male_df =  df[df[3]==1]
                    female_df = df[df[3]==0]
                    end_ind = len(female_df) // 3
                    df = male_df
            if test:
                test_mode = config.config["test_mode"]
                if test_mode == 1:
                    df = df[df[3] == 1]
                elif test_mode == 2:
                    df = df[df[3] == 0]
            # Do verification not gender classification

            print("Male : {}, Female: {}".format(len(df[df[3] ==1]), len(df[df[3]==0])))

            class_type = config.config["class"]
            if class_type == "gender":
                y = df[3]
            elif class_type == "age":
                y = df[2]
            elif class_type == "subject":
                y = df[0].apply(lambda x: int(x[1:5]))
            if self.type == "aligned":
                df[0] = df[0].apply(lambda x: x.split('.')[0][:-1] + '.jpg')
            df[0] = df[0].apply(lambda x: os.path.join(self.train_dir, x))
            
            # y = df[3] if class_type == 'gender' else df[2]
            x = df[0]

            
            list_ds = tf.data.Dataset.from_tensor_slices(x)
            image_count = (len(list_ds))

            


            list_ds = tf.data.Dataset.zip(
                (list_ds, tf.data.Dataset.from_tensor_slices(y)))
            if not test:
                list_ds = list_ds.shuffle(
                    image_count, reshuffle_each_iteration=False)
            return list_ds

        fold = config.config["fold"]
        self.train_ds = get_data(r'\train_fold' + fold + "*")
        self.val_ds = get_data(r"\val_fold" + fold + "*")
        self.test_ds = get_data(r"\test_fold" + fold + "*", True)

        if verbose:
            for f in self.train_ds.take(5):
                print(f[0].numpy(), f[1].numpy())


        self.train_ds = self.train_ds.map(
            lambda x, y: self.process_path((x, y), True), num_parallel_calls=self.AUTOTUNE)
        self.val_ds = self.val_ds.map(lambda x, y: self.process_path(
            (x, y), True), num_parallel_calls=self.AUTOTUNE)
        self.test_ds = self.test_ds.map(lambda x, y: self.process_path(
            (x, y), False), num_parallel_calls=self.AUTOTUNE)

        if verbose:
            for image, label in self.train_ds.take(1):
                print("Image shape: ", image.numpy().shape)
                print("Label: ", label.numpy())
        
        train_ds = tf.data.Dataset.concatenate(self.train_ds, self.val_ds)
        self.val_ds = self.test_ds

        image_count = (len(self.train_ds))
        val_size = int(image_count * 0.2)
        self.train_ds = train_ds.skip(val_size)
        self.val_ds = train_ds.take(val_size)

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
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        ds = ds.with_options(options)
        ds = ds.cache()
        if training:
            ds = ds.shuffle(buffer_size=self.buffer_size)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)
        return ds
