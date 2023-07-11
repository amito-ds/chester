import numpy as np
import os
import random
import numpy as np
import pandas as pd
# exec(open("/Users/amitosi/PycharmProjects/chester/diamond/manual_run.py").read())
from PIL import Image

from diamond.run import run
from diamond.user_classes import ImagesAugmentationInfo, ImageModel, ImageModels

# # define directory path
# data_dir = '/Users/amitosi/PycharmProjects/chester/chester/data/pics_ex'
# data_dir = '/Users/amitosi/PycharmProjects/chester/chester/data/pics_ex2'
data_dir = '/Users/amitosi/PycharmProjects/chester/chester/data/pic_ex3'

# # define image size
img_size = (1024, 1024)
# img_size = (640, 480)
# img_size = (32, 32)
#
# define empty arrays to store images and labels
images = []
labels = []
# Loop through each file in the pizza folder and add it to the images list with label 0 (pizza)
for filename in os.listdir(data_dir):
    image = Image.open(os.path.join(data_dir, filename))

    # # Define the area of the image to zoom in on
    # left = 50
    # right = 4000
    # top = left
    # bottom = right
    #
    # # Crop the image to the defined area
    # cropped_img = image.crop((left, top, right, bottom))
    #
    # # Resize the cropped image to make it appear zoomed in
    # zoomed_img = cropped_img.resize((600, 600))
    #
    # # Show the zoomed image
    # # zoomed_img.show()

    # images.append(zoomed_img)
    images.append(image)

# images = random.sample(images, 7)

new_images = []

# for image in images:
#     img = image.resize(img_size)
#     img_array = np.array(img)
#     new_images.append(img_array)

# new_images = np.array(new_images)
print("Total pics", len(new_images))

# labels = pd.read_csv(data_dir + "/labels.csv").fillna('UNK')['Category']

# # convert the lists to NumPy arrays
# images = np.array(images[0:100])
# print(np.unique(labels))
# labels = np.array(labels[0:100])

image_shape = (3, 1024, 1024)
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
                        # labels=labels,
                        # get_image_description=True,
                        get_object_detection=True,
                        # detect_faces=True,
                        is_augment_data=False,
                        image_augmentation_info=ImagesAugmentationInfo(aug_prop=0.7),
                        is_train_model=False, image_models=image_models,
                        is_post_model_analysis=False,
                        plot=True)
