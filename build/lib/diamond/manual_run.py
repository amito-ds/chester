import math
import pickle
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
# exec(open("/Users/amitosi/PycharmProjects/chester/diamond/manual_run.py").read())
from PIL import Image

from diamond.run import run
from diamond.user_classes import ImageModels, ImageModel, ImagesAugmentationInfo
from sklearn.datasets import fetch_openml

with open("/Users/amitosi/PycharmProjects/chester/chester/data/data_batch_1", 'rb') as fo:
    data = pickle.load(fo, encoding='bytes')
# Extract the image and label data from the batch


# Load the MNIST dataset
mnist = fetch_openml('mnist_784')
# Extract the data and labels
X, y = mnist['data'], mnist['target']

X = X.astype(np.float32)
# X.to_csv('mnist_fashion_X.csv', index=False)
# y.to_csv('mnist_fashion_y.csv', index=False)


image_model_list = [
    ImageModel(network_name="EfficientNetB0",
               batch_size=64 * 64 * 16,
               num_epochs=1,
               optimizer_params={'lr': 0.005},
               dropout=0.7),

    # ImageModel(network_name="ResNet101", batch_size=64 * 2, num_epochs=8, optimizer_params={'lr': 0.05},
    #            dropout=0.5),
    #
    # ImageModel(network_name="ResNet101", batch_size=64 * 2, num_epochs=6, optimizer_params={'lr': 0.05},
    #            dropout=0.2),
    # ImageModel(network_name="EfficientNetB0", batch_size=32 * 32, num_epochs=5, optimizer_params={'lr': 1})
]
image_models = ImageModels(image_model_list=image_model_list)

# y = pd.read_csv('/Users/amitosi/PycharmProjects/chester/mnist_fashion_y.csv')
# y = y['class']
# X = pd.read_csv('/Users/amitosi/PycharmProjects/chester/mnist_fashion_X.csv')

image_shape = (28, 28)
# image_shape = (32, 32)
diamond_collector = run(images=X[0:100],
                        image_shape=image_shape,
                        labels=y[0:100],
                        get_image_description=False,
                        is_augment_data=False,
                        detect_faces=False,
                        image_augmentation_info=ImagesAugmentationInfo(aug_prop=0.2),
                        is_train_model=True, image_models=image_models,
                        is_post_model_analysis=True,
                        plot=False)

# labels handling
# image convert to np nd array
#
# images = data[b'data'][0:100]
# labels = np.array(data[b'labels'])[0:2500]
# labels = data[b'labels'][0:100]
# image_shape = (3, 32, 32)
#
# diamond_collector = run(images=images,
#                         image_shape=image_shape,
#                         labels=labels,
#                         get_image_description=True,
#                         is_augment_data=False,
#                         image_augmentation_info=ImagesAugmentationInfo(aug_prop=0.01),
#                         is_train_model=False, image_models=image_models,
#                         is_post_model_analysis=False,
#                         plot=False)

# print(diamond_collector["models"])
#
#

# #
# # # define directory path
# data_dir = '/Users/amitosi/PycharmProjects/chester/chester/data/weather'
#
# # # define image size
# img_size = (64, 64)
# # img_size = (640, 480)
# # img_size = (32, 32)
# #
# # define empty arrays to store images and labels
# images = []
# labels = []
#
# # loop over the directories containing the images
#
# # loop over the files in the directory
# for filename in os.listdir(data_dir):
#     if filename.endswith('.jpg'):
#         label = filename.split("/")[-1].split(".")[0]
#         label = ''.join(char for char in label if not char.isdigit())
#         # read the image as a PIL.Image object
#         img = Image.open(os.path.join(data_dir, filename)).convert('RGB')
#         # print(img.size)
#         # resize the image
#         img = img.resize(img_size)
#         # img = img.resize(img.size)
#         # convert the image to a NumPy array
#         img_array = np.array(img)
#         # append the image array to the list of images
#         images.append(img_array)
#         # append the label to the list of labels
#         labels.append(label)
#
# # # convert the lists to NumPy arrays
# images = np.array(images[0:50])
# # print(np.unique(labels))
# labels = np.array(labels[0:50])
#
# image_shape = (3, 64, 64)
# # image_shape = (3, 480, 640)
# #
# diamond_collector = run(images=images,
#                         image_shape=image_shape,
#                         labels=labels,
#                         get_image_description=True,
#                         is_augment_data=False,
#                         image_augmentation_info=ImagesAugmentationInfo(aug_prop=0.1),
#                         is_train_model=False, image_models=image_models,
#                         is_post_model_analysis=False,
#                         plot=True)
