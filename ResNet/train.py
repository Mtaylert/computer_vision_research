import config
import model
import datasetup


def run():
    imnorm = datasetup.ImageNormalization(
        train_image_folder="../data/intel/seg_train/seg_train/",
        test_image_folder="../data/intel/seg_test/seg_test/",
    )
    cache = imnorm.normalize()
    no_of_classes = len(cache["train"].classes)
    device = config.get_default_device()
    resnet = config.to_device(model.ResNet9(3, no_of_classes), device)
    print(resnet)


if __name__ == "__main__":
    run()
