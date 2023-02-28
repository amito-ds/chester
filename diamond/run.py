import random
import numpy as np
from diamond.data_augmentation.augmentation import ImageAugmentation
from diamond.image_data_info.image_info import ImageInfo
from diamond.model_training.best_model import ImageModelsTraining
from diamond.user_classes import ImagesData, ImagesAugmentationInfo, ImageModels


def run(images,
        labels=None,
        validation_prop=0.2,
        is_augment_data=True, image_augmentation_info=None,
        is_train_model=True, image_models=None,
        is_post_model_analysis=True,
        plot=True
        ):
    diamond_collector = {}

    # load the data
    images = images.reshape(-1, 3, 32, 32)

    if labels is None:
        is_train_model = False
        pass  # TODO: create labels of 1s

    # Image data
    image_data = ImagesData(images=images, labels=labels, validation_prop=validation_prop)
    diamond_collector["image_data"] = image_data
    # plot
    print("Sample Plot")
    if plot:
        image_data.plot_images()

    # augmentation
    if is_augment_data:
        if image_augmentation_info is None:
            image_augmentation_info = ImagesAugmentationInfo()
        image_data = ImageAugmentation(image_data, image_augmentation_info).run()
        diamond_collector["augmented_image_data"] = image_data
        # plot
        print("Updated Sample Plot")
        if plot:
            image_data.plot_images()

    # Training
    if not is_train_model:
        return diamond_collector
    if image_models is None:
        image_models = ImageModels()
    best_models = ImageModelsTraining(images_data=image_data,
                                      image_models=image_models).run()  # return the model ordered by best to worst

    diamond_collector["models"] = best_models

    # Post model analysis
    if is_post_model_analysis:
        pass

    return diamond_collector
