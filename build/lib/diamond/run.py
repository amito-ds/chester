import random
import numpy as np
from diamond.data_augmentation.augmentation import ImageAugmentation
from diamond.face_detection.face_image_ import ImageFaceDetection
from diamond.image_caption.image_caption_class import ImageDescription
from diamond.image_data_info.image_info import ImageInfo
from diamond.image_object_detection.image_object_class import ImageObjectDetection
from diamond.model_training.best_model import ImageModelsTraining
from diamond.post_model_analysis.post_model import ImagePostModelAnalysis
from diamond.user_classes import ImagesData, ImagesAugmentationInfo, ImageModels, ImagePostModelSpec, \
    ImageDescriptionSpec
import pandas as pd

from diamond.utils import index_labels


def run(images,
        image_shape,
        labels=None,
        validation_prop=0.3,
        get_image_description=False, image_description_spec: ImageDescriptionSpec = None,
        get_object_detection=False,
        detect_faces=False,
        is_augment_data=True, image_augmentation_info=None,
        is_train_model=True, image_models=None,
        is_post_model_analysis=True, image_post_model: ImagePostModelSpec = None,
        plot=True, plot_sample=20
        ):
    # Tell a story
    story = """Welcome to MadCat, the comprehensive machine learning and data analysis solution!
    \nThis module is designed to streamline the entire process of video tasks,
    \nfrom start to finish.
    \nTo learn more about MadCat, visit https://github.com/amito-ds/chester.\n"""
    print(story)
    diamond_collector = {}

    if labels is None:
        is_train_model = False
        labels = pd.Series([1] * len(images))
    # Image data
    if type(images).__name__ == "DataFrame":
        images = images.values

    label_dict, labels = index_labels(labels)

    image_data = ImagesData(images=images, labels=labels,
                            validation_prop=validation_prop,
                            image_shape=image_shape,
                            label_dict=label_dict,
                            for_model_training=is_train_model)
    diamond_collector["image_data"] = image_data
    # plot
    if plot:
        print("Plotting Sample of Images")
        image_data.plot_images(plot_sample=plot_sample)

    # Image description
    if get_image_description:
        print("====> Calculating Image Description")
        if image_description_spec is None:
            image_description_spec = ImageDescriptionSpec()
        ImageDescription(
            images_data=image_data,
            image_description_spec=image_description_spec,
            diamond_collector=diamond_collector,
            plot=plot).run()
    # Object detection
    if get_object_detection:
        print("====> Detecting Objects in the Images")
        ImageObjectDetection(
            images_data=image_data,
            diamond_collector=diamond_collector,
            plot_sample=plot_sample,
            plot=plot).run()

    # Face detection
    if detect_faces:
        print("====> Detecting Faces in the Images")
        ImageFaceDetection(
            images_data=image_data,
            diamond_collector=diamond_collector,
            plot_sample=plot_sample,
            plot=plot).run()

    # augmentation
    if is_augment_data:
        print("====> Augmenting data")
        if image_augmentation_info is None:
            image_augmentation_info = ImagesAugmentationInfo()
        image_data = ImageAugmentation(image_data, image_augmentation_info).run()
        diamond_collector["augmented_image_data"] = image_data
        # plot
        if plot:
            print("Updated Plotting Sample of Images")
            image_data.plot_images(plot_sample=plot_sample)

    # Training
    if not is_train_model:
        return diamond_collector
    if image_models is None:
        image_models = ImageModels()
    s = ''
    if image_models.n_models > 1:
        s = 's'
    print(f"====> Training {image_models.n_models} Model{s}")
    models_sorted = ImageModelsTraining(images_data=image_data,
                                        image_models=image_models).run()  # return the model ordered by best to worst

    diamond_collector["models"] = models_sorted

    # Post model analysis
    if is_post_model_analysis:
        print("====> Analyzing the best model")
        if image_post_model is None:
            image_post_model = ImagePostModelSpec(plot=plot)
        ImagePostModelAnalysis(model_list=models_sorted,
                               images_data=image_data,
                               image_post_model=image_post_model,
                               diamond_collector=diamond_collector,
                               plot=plot).run()

    return diamond_collector
