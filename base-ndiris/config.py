config = {
    "run_id": "",
    "model": "ResNet50",
    "mixed_precision": True,
    # "run_id": "DenseNet121_UFPR-aligned_gender",
    "class": "gender",
    "allow_growth": True,
    "mode": "train",
    "viz_method": "gen_data",
    "test_mode": 0, # 0, 1, 2 -> 50-50, Male only, Female only Dont change here
    "fine-tune": True,
    "ratio": "F50M50",
    "fold": "1", # Dont change here
    "optimizer": {
        "name": "adam",
        "initial_learning_rate": 3e-4,
        "lr_scheduler": "none",
        "first_decay_steps": 1000,
        "decay_rate": 0.9,
    },
    "train": {
        "batch_size": 1024,
        "gpu": [0, 1],
        "epochs": 100,
        "unfreeze_epoch": 1000,
        "steps_per_batch": 100,
        "validation_split": 0.2,
        "buffer_size": 1024,
        "ckpt_dir": r"F:\Lab\nfs\base-ndiris\checkpoints",
        'pretrained': False,
        'pretrained_weight': r"F:\Lab\nfs\base-visob\checkpoints\ResNet50_UFPR-aligned-cropped_aug_gender.07-0.4004.hdf5",
        'pretrained_epoch': 15,
    },
    "input": {
        "type": "band", # left, right or band
        "image_height": 168, # 532//3,
        "image_width": 224, # 1470//3,
        "channels": 3,
        "train_dir":  r"F:\Lab\datasets\nd-iris\train.tfrecord",
        "eval_dir":  r"F:\Lab\datasets\nd-iris\test.tfrecord",
        "labels": r"F:\Lab\nfs\base-notredame\nir_labels_",
        "randaugment": True,
        "randaugment_layers": 2,
        "randaugment_magnitude": 12
    }
}
