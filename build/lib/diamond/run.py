import random
import numpy as np
from diamond.data_augmentation.augmentation import ImageAugmentation
from diamond.image_data_info.image_info import ImageInfo
from diamond.model_training.best_model import ImageModelsTraining
from diamond.post_model_analysis.post_model import ImagePostModelAnalysis
from diamond.user_classes import ImagesData, ImagesAugmentationInfo, ImageModels, ImagePostModelSpec


def run(images,
        image_shape,
        labels=None,
        validation_prop=0.3,
        is_augment_data=True, image_augmentation_info=None,
        is_train_model=True, image_models=None,
        is_post_model_analysis=True, image_post_model: ImagePostModelSpec = None,
        plot=True
        ):
    # Tell a story
    story = """Welcome to MadCat, the comprehensive machine learning and data analysis solution!
    \nThis module is designed to streamline the entire process image classification models, 
    \nfrom start to finish.
    \nTo learn more about MadCat, visit https://github.com/amito-ds/chester.\n"""
    print(story)
    diamond_collector = {}

    if labels is None:
        is_train_model = False
        pass  # TODO: create labels of 1s

    # Image data
    image_data = ImagesData(images=images, labels=labels, validation_prop=validation_prop, image_shape=image_shape)
    diamond_collector["image_data"] = image_data
    # plot
    if plot:
        print("Sample Plot")
        image_data.plot_images()

    # augmentation
    if is_augment_data:
        print("====> Augmenting data")
        if image_augmentation_info is None:
            image_augmentation_info = ImagesAugmentationInfo()
        image_data = ImageAugmentation(image_data, image_augmentation_info).run()
        diamond_collector["augmented_image_data"] = image_data
        # plot
        if plot:
            print("Updated Sample Plot")
            image_data.plot_images()

    # Training
    if not is_train_model:
        print("====> Training a Model")
        return diamond_collector
    if image_models is None:
        image_models = ImageModels()
    models_sorted = ImageModelsTraining(images_data=image_data,
                                        image_models=image_models).run()  # return the model ordered by best to worst

    diamond_collector["models"] = models_sorted

    # Post model analysis
    if is_post_model_analysis:
        if image_post_model is None:
            image_post_model = ImagePostModelSpec(plot=plot)
        ImagePostModelAnalysis(model_list=models_sorted,
                               images_data=image_data,
                               image_post_model=image_post_model,
                               diamond_collector=diamond_collector,
                               plot=plot).run()

    return diamond_collector
