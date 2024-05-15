config = {
    "run_id": "",
    "model": "ResNet50",
    "mixed_precision": False,
    "min_diff": True,
    "min_diff_weight": 1.5,
    # "run_id": "DenseNet121_UFPR-aligned_gender",
    "class": "gender",
    "allow_growth": True,
    "mode": "train",
    "viz_method": "gen_data",
    "test_mode": 1, # 0, 1, 2 -> 50-50, Male only, Female only Dont change here
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
        "batch_size": 128,
        "gpu": [0, 1],
        "epochs": 100,
        "unfreeze_epoch": 1000,
        "steps_per_batch": 100,
        "validation_split": 0.2,
        "buffer_size": 1024,
        "ckpt_dir": r"F:\Lab\nfs\base-fairface\checkpoints",
        'pretrained': False,
        'pretrained_weight': r"F:\Lab\nfs\base-fairface\checkpoints\ResNet50_UFPR-aligned-cropped_aug_gender.07-0.4004.hdf5",
        'pretrained_epoch': 15,
    },
    "input": {
        "type": "cropped",
        "image_height": 224,
        "image_width": 224,
        "channels": 3,
        "train_dir":  r"F:\Lab\datasets\fairface\fairface-img-margin025-trainval",
        "eval_dir":  r"F:\Lab\datasets\fairface\fairface-img-margin025-trainval",
        "labels": r"F:\Lab\datasets\fairface",
        "randaugment": True,
        "randaugment_layers": 2,
        "randaugment_magnitude": 8
    }
}
