import pickle

import numpy as np

from diamond.run import run
from diamond.user_classes import ImageModels, ImageModel, ImagesAugmentationInfo

with open("/Users/amitosi/PycharmProjects/chester/chester/data/data_batch_1", 'rb') as fo:
    data = pickle.load(fo, encoding='bytes')

# Extract the image and label data from the batch
images = data[b'data'][0:2500]
labels = np.array(data[b'labels'])[0:2500]

image_model_list = [
    ImageModel(network_name="EfficientNetB0",
               batch_size=64 * 32,
               num_epochs=2,
               optimizer_params={'lr': 0.05},
               dropout=0.4),

    # ImageModel(network_name="ResNet101", batch_size=64 * 2, num_epochs=8, optimizer_params={'lr': 0.05},
    #            dropout=0.5,
    #            remove_num_layers_layers=1),
    #
    # ImageModel(network_name="ResNet101", batch_size=64 * 2, num_epochs=6, optimizer_params={'lr': 0.05},
    #            dropout=0.2,
    #            remove_num_layers_layers=1),
    # ImageModel(network_name="EfficientNetB0", batch_size=32 * 32, num_epochs=2, optimizer_params={'lr': 1})
]
image_models = ImageModels(image_model_list=image_model_list)

diamond_collector = run(images,
                        labels,
                        is_augment_data=False,
                        image_augmentation_info=ImagesAugmentationInfo(aug_prop=0.01),
                        is_train_model=True, image_models=image_models,
                        is_post_model_analysis=True,
                        plot=False)

# print(diamond_collector["models"])
