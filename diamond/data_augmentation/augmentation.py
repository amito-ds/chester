import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

from diamond.user_classes import ImagesData, ImagesAugmentationInfo
from scipy.ndimage import rotate, zoom

import math


class ImageAugmentation:
    def __init__(self, images_data: ImagesData, image_augmentation_info: ImagesAugmentationInfo):
        self.images_data = images_data
        self.image_augmentation_info = image_augmentation_info
        # cals
        self.images, self.labels = self.images_data.images, self.images_data.labels
        self.aug_types, self.aug_prop = self.image_augmentation_info.aug_types, self.image_augmentation_info.aug_prop
        self.is_colored = self.images_data.is_colored
        self.image_shape = self.images_data.image_shape
        self.label_dict = self.images_data.label_dict

    def rotate_image(self):
        # Sample images to rotate
        num_images = len(self.images)
        num_to_rotate = math.ceil(num_images * self.aug_prop)
        image_indices = random.choices(range(num_images), k=num_to_rotate)

        # Apply rotation to sampled images
        if self.is_colored:
            rotated_images = []
            for i, index in enumerate(image_indices):
                image = self.images[index]
                # image = np.transpose(image, (1, 2, 0))
                angle = random.randint(-180, 180)
                rotated = rotate(image, angle, reshape=True)
                resized_rotated = cv2.resize(rotated, dsize=image.shape[:2][::-1])
                rotated_images.append(resized_rotated)
            # Concatenate rotated images with original images
            try:
                self.images = np.concatenate((self.images, rotated_images), axis=0)
            except:
                self.images = np.concatenate((self.images, rotated_images))
            self.labels = np.concatenate((self.labels, self.labels[image_indices]), axis=0)
            # Check that the number of images and labels is the same
            if len(self.images) != len(self.labels):
                raise ValueError(f"Number of images and labels must be the same, {len(self.images), len(self.labels)}")
        else:
            rotated_images = []
            for i, index in enumerate(image_indices):
                image = self.images[index]
                angle = random.randint(-180, 180)
                rotated = rotate(image, angle, reshape=True)
                rotated = cv2.resize(rotated, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
                rotated_images.append(rotated)

            # Concatenate rotated images with original images
            try:
                self.images = np.concatenate((self.images, rotated_images), axis=0)
            except:
                self.images = np.concatenate((self.images, rotated_images))
            self.labels = np.concatenate((self.labels, self.labels[image_indices]), axis=0)
            # Check that the number of images and labels is the same
            if len(self.images) != len(self.labels):
                raise ValueError(f"Number of images and labels must be the same, {len(self.images), len(self.labels)}")

    from scipy.ndimage import zoom

    def zoom_image(self):
        if self.is_colored:
            # Retrieve size and number of channels of the first image in the data
            try:
                orig_height, orig_width = self.image_shape
            except:
                _, orig_height, orig_width = self.image_shape
            # Sample images to zoom
            num_images = len(self.images)
            num_to_zoom = int(num_images * self.aug_prop)
            image_indices = random.choices(range(num_images), k=num_to_zoom)

            # Apply zooming to sampled images
            zoomed_images = np.zeros_like(self.images[:num_to_zoom])
            for i, index in enumerate(image_indices):
                image = self.images[index]
                # Resize the image while maintaining the aspect ratio
                eps = 0.2
                zoomed = zoom(image, (random.uniform(eps, 2 - eps), random.uniform(eps, 2 - eps), 1))
                # Crop or pad the zoomed image to maintain the original shape
                new_height, new_width, _ = zoomed.shape
                if new_height > orig_height:
                    crop_top = (new_height - orig_height) // 2
                    crop_bottom = new_height - orig_height - crop_top
                    zoomed = zoomed[crop_top:new_height - crop_bottom, :, :]
                else:
                    pad_top = (orig_height - new_height) // 2
                    pad_bottom = orig_height - new_height - pad_top
                    zoomed = np.pad(zoomed, ((pad_top, pad_bottom), (0, 0), (0, 0)), 'constant')
                if new_width > orig_width:
                    crop_left = (new_width - orig_width) // 2
                    crop_right = new_width - orig_width - crop_left
                    zoomed = zoomed[:, crop_left:new_width - crop_right, :]
                else:
                    pad_left = (orig_width - new_width) // 2
                    pad_right = orig_width - new_width - pad_left
                    if self.is_colored:
                        zoomed = np.pad(zoomed, ((0, 0), (pad_left, pad_right), (0, 0)), 'constant')
                    else:
                        zoomed = np.pad(zoomed, ((0, 0), (pad_left, pad_right)), 'constant')
                # Convert the image to the original data type
                zoomed = zoomed.astype(image.dtype)
                zoomed = cv2.resize(zoomed, dsize=image.shape[:2][::-1])
                zoomed_images[i] = zoomed

            # Concatenate zoomed images with original images
            self.images = np.concatenate((self.images, zoomed_images), axis=0)
            self.labels = np.concatenate((self.labels, self.labels[image_indices]), axis=0)

            # Check that the number of images and labels is the same
            if len(self.images) != len(self.labels):
                raise ValueError(f"Number of images and labels must be the same, {len(self.images), len(self.labels)}")
        else:
            # Retrieve size and number of channels of the first image in the data
            orig_height, orig_width = self.image_shape
            # Sample images to zoom
            num_images = len(self.images)
            num_to_zoom = int(num_images * self.aug_prop)
            image_indices = random.choices(range(num_images), k=num_to_zoom)

            # Apply zooming to sampled images
            zoomed_images = np.zeros_like(self.images[:num_to_zoom])
            for i, index in enumerate(image_indices):
                image = self.images[index]
                # Resize the image while maintaining the aspect ratio
                zoomed = zoom(image, (random.uniform(0.2, 1.8), random.uniform(0.2, 1.8)))
                # Crop or pad the zoomed image to maintain the original shape
                new_height, new_width = zoomed.shape
                if new_height > orig_height:
                    crop_top = (new_height - orig_height) // 2
                    crop_bottom = new_height - orig_height - crop_top
                    zoomed = zoomed[crop_top:new_height - crop_bottom, :]
                else:
                    pad_top = (orig_height - new_height) // 2
                    pad_bottom = orig_height - new_height - pad_top
                    zoomed = np.pad(zoomed, ((pad_top, pad_bottom), (0, 0)), 'constant')
                if new_width > orig_width:
                    crop_left = (new_width - orig_width) // 2
                    crop_right = new_width - orig_width - crop_left
                    zoomed = zoomed[:, crop_left:new_width - crop_right]
                else:
                    pad_left = (orig_width - new_width) // 2
                    pad_right = orig_width - new_width - pad_left
                    zoomed = np.pad(zoomed, ((0, 0), (pad_left, pad_right)), 'constant')
                # Convert the image to the original data type
                zoomed = zoomed.astype(image.dtype)
                zoomed_images[i] = zoomed

            # Concatenate zoomed images with original images
            self.images = np.concatenate((self.images, zoomed_images), axis=0)
            self.labels = np.concatenate((self.labels, self.labels[image_indices]), axis=0)

            # Check that the number of images and labels is the same
            if len(self.images) != len(self.labels):
                raise ValueError(f"Number of images and labels must be the same, {len(self.images), len(self.labels)}")

    def run(self):
        if "rotate" in self.aug_types:
            self.rotate_image()
        if "zoom" in self.aug_types:
            self.zoom_image()

        images_data = ImagesData(images=self.images, labels=self.labels,
                                 validation_prop=self.images_data.validation_prop,
                                 image_shape=self.images_data.image_shape,
                                 label_dict=self.label_dict,
                                 for_model_training=self.images_data.for_model_training)
        self.images_data = images_data
        return images_data
