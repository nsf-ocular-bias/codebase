import config
import main
import glob
import os


def setup_config():
    model = config.config["model"]
    class_type = config.config["class"]
    finetune = "finetuned" if config.config["fine-tune"] else "base"
    input_type = config.config["input"]["type"]
    augmented = "aug" if config.config["input"]["randaugment"] else "noaug"
    ratio = config.config["ratio"]

    run_id = '_'.join([model, "Notredame", input_type,
                      augmented, class_type, finetune, ratio])
                      
    # run_id = '_'.join([model, "UFPR", input_type,
    #                   augmented, class_type, finetune])
    config.config["run_id"] = run_id


def run():
    models = ["EfficientNetB4", "DenseNet121", "MobileNetV2", "InceptionV3",
              "InceptionResNetV2", "ResNet50", "VGG19", "Xception", "EfficientNetB0"]

    # models = ["EfficientNetB4", "DenseNet121", "MobileNetV2", "InceptionV3",
    #           "InceptionResNetV2", "ResNet50", "Xception", "EfficientNetB0"]
    ckpt_dir = config.config["train"]["ckpt_dir"]


    config.config["input"]["randaugment"] = True

    
    
    config.config["mode"] = "test"

    class_type = "gender"
    config.config["class"] = class_type
    config.config["test_mode"] = 2
    weight_dir = ckpt_dir
    config.config["fine-tune"] = True
    for model in models:
        config.config["model"] = model
        setup_config()
        run_id = config.config["run_id"]
        print(run_id)
        files = glob.glob(os.path.join(
            weight_dir, config.config["run_id"] + "*"))
        files.sort()
        # print(files)
        print(files[-1])
        config.config["train"]["pretrained_weight"] = files[-1]
        main.main()



  
    # for class_type in ["gender"]:
    #     config.config["class"] = class_type

    #     # run base models
    #     config.config["fine-tune"] = False
    #     for model in models:
    #         config.config["model"] = model
    #         setup_config()
    #         main.main()

    #     # Run fine tuned
    #     config.config["fine-tune"] = True
    #     for model in models:
    #         config.config["model"] = model
    #         setup_config()
    #         # set pretrained weights
    #         files = glob.glob(os.path.join(
    #             ckpt_dir, config.config["run_id"].replace("finetuned", "base") + "*"))
    #         files.sort()
    #         config.config["train"]["pretrained_weight"] = files[-1]
    #         main.main()


if __name__ == "__main__":
    run()
