import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import config
import numpy as np
import model
import random

def main():
    print("Running...", config.config)
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)
    # Set Memory Growth
    if config.config["allow_growth"]:
        config_proto = tf.compat.v1.ConfigProto()
        config_proto.gpu_options.allow_growth = True
        session = tf.compat.v1.Session(config=config_proto)
        tf.compat.v1.keras.backend.set_session(session)

    # Set GPU visibility
    gpus = config.config["train"]["gpu"]
    if gpus == [0, 1]:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
    epochs = config.config["train"]["epochs"]
    unfreeze_epoch = config.config['train']['unfreeze_epoch']
    pretrained = config.config['train']['pretrained']

    

    _model = model.Module()
    initial_epoch = 0
    weight = config.config['train']['pretrained_weight']
    if pretrained:
        _model.load_pretrained(weight)
        initial_epoch = config.config['train']['pretrained_epoch']
    
    mode = config.config['mode']
    if mode == 'train':
        _model.train_model(epochs=epochs, unfreeze_epoch=unfreeze_epoch, initial_epoch=initial_epoch)
    elif mode == 'test':
        finetune = config.config["fine-tune"]
        print(finetune)
        if finetune:
            _model.unfreeze_model()
        _model.load_pretrained(weight)
        _model.test_model()
    elif mode == 'viz':
        finetune = config.config["fine-tune"]
        if finetune:
            _model.unfreeze_model()
        _model.load_pretrained(weight)
        _model.viz_model()

if __name__ == '__main__':
    main()
