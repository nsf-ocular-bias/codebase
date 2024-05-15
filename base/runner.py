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

    run_id = '_'.join([model, "UFPR", input_type,
                      augmented, class_type, finetune, ratio])
                      
    # run_id = '_'.join([model, "UFPR", input_type,
    #                   augmented, class_type, finetune])
    config.config["run_id"] = run_id


def run():
    models = ["EfficientNetB4", "DenseNet121", "MobileNetV2", "InceptionV3",
              "InceptionResNetV2", "ResNet50", "VGG19", "Xception", "EfficientNetB0"]
    # models = ["EfficientNetB4", "DenseNet121", "MobileNetV2", "InceptionV3",
    #         "InceptionResNetV2", "ResNet50", "Xception", "EfficientNetB0"]
    ckpt_dir = config.config["train"]["ckpt_dir"]

    input_type = "aligned"

    config.config["input"]["randaugment"] = True

    config.config["input"]["type"] = input_type
    if input_type == "aligned":
        config.config["input"]["image_height"] = 256
        config.config["input"]["image_width"] = 512
    else:
        config.config["input"]["image_height"] = 256
        config.config["input"]["image_width"] = 256
    
    config.config["mode"] = "viz"
    class_type = "subject"
    config.config["class"] = class_type



    config.config["test_mode"] = 0
    # weight_dir = os.path.join(ckpt_dir, world)
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
        config.config["train"]["pretrained_weight"] = files[-1]
        print(files[-1])
        main.main()



    # for i,world in enumerate([r"F:\Lab\datasets\UFPR-Periocular\UFPR-Periocular\experimentalProtocol\open_world_valopen",
    #               r"F:\Lab\datasets\UFPR-Periocular\UFPR-Periocular\experimentalProtocol\closed_world"]):
        
    #     if i == 1:
    #         continue
    #     config.config["input"]["labels"] = world
    #     for class_type in ["subject"]:
    #         config.config["class"] = class_type

    #         # run base models
    #         config.config["fine-tune"] = False
    #         for model in models:
    #             config.config["model"] = model
    #             setup_config()
    #             main.main()

    #         # Run fine tuned
    #         config.config["fine-tune"] = True
    #         for model in models:
    #             config.config["model"] = model
    #             setup_config()
    #             # set pretrained weights
    #             files = glob.glob(os.path.join(
    #                 ckpt_dir, config.config["run_id"].replace("finetuned", "base") + "*"))
    #             files.sort()
    #             config.config["train"]["pretrained_weight"] = files[-1]
    #             main.main()


if __name__ == "__main__":
    run()
