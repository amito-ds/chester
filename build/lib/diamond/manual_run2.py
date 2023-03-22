import numpy as np
import os

import numpy as np
import pandas as pd
# exec(open("/Users/amitosi/PycharmProjects/chester/diamond/manual_run.py").read())
from PIL import Image

from diamond.run import run
from diamond.user_classes import ImagesAugmentationInfo, ImageModel, ImageModels

# # define directory path
data_dir = '/Users/amitosi/PycharmProjects/chester/chester/data/blood_type'

# # define image size
img_size = (64, 64)
# img_size = (640, 480)
# img_size = (32, 32)
#
# define empty arrays to store images and labels
images = []
labels = []

# loop over the directories containing the images

# loop over the files in the directory
for filename in os.listdir(data_dir):
    if filename.endswith('.jpg'):
        # label = filename.split("/")[-1].split(".")[0]
        # label = ''.join(char for char in label if not char.isdigit())
        # read the image as a PIL.Image object
        img = Image.open(os.path.join(data_dir, filename)).convert('RGB')
        # print(img.size)
        # resize the image
        img = img.resize(img_size)
        # img = img.resize(img.size)
        # convert the image to a NumPy array
        img_array = np.array(img)
        # append the image array to the list of images
        images.append(img_array)
        # append the label to the list of labels

labels = pd.read_csv(data_dir + "/labels.csv").fillna('UNK')['Category']

# # convert the lists to NumPy arrays
# images = np.array(images[0:100])
# print(np.unique(labels))
images = images[0:20]
labels = np.array(labels[0:20])

image_shape = (3, 64, 64)
# image_shape = (3, 480, 640)


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
                        detect_faces=False,
                        get_object_detection=False,
                        is_augment_data=False,
                        image_augmentation_info=ImagesAugmentationInfo(aug_prop=0.1),
                        is_train_model=True, image_models=image_models,
                        is_post_model_analysis=True,
                        plot=True)
