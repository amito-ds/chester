import pickle

import numpy as np

from diamond.run import run
from diamond.user_classes import ImagesAugmentationInfo, ImageModel, ImageModels

# exec(open("/Users/amitosi/PycharmProjects/chester/diamond/manual_run.py").read())

# define directory path
with open("/Users/amitosi/PycharmProjects/chester/chester/data/data_batch_1", 'rb') as fo:
    data = pickle.load(fo, encoding='bytes')
# subdir: not pizza, .jpg
# subdir: pizza, .jpg

# convert the images and labels arrays to numpy arrays
# Extract the images and labels
images = data[b'data'][0:100]
labels = np.array(data[b'labels'][0:100])

# images = np.array(images[0:50])
# labels = np.array(labels[0:50])


image_shape = (3, 32, 32)

image_model_list = [
    ImageModel(network_name="EfficientNetB0",
               batch_size=64 * 64 * 16,
               num_epochs=1,
               optimizer_params={'lr': 0.005},
               dropout=0.7)]
image_models = ImageModels(image_model_list=image_model_list)

diamond_collector = run(images=images,
                        image_shape=image_shape,
                        labels=labels,
                        get_image_description=False,
                        get_object_detection=False,
                        detect_faces=False,
                        is_augment_data=True,
                        image_augmentation_info=ImagesAugmentationInfo(aug_prop=0.1),
                        is_train_model=True, image_models=image_models,
                        is_post_model_analysis=True,
                        plot=True)
