import cv2
import random
import numpy as np

from diamond.user_classes import ImagesData, ImagesAugmentationInfo


class ImageAugmentation:
    def __init__(self, images_data: ImagesData, image_augmentation_info: ImagesAugmentationInfo):
        self.images_data = images_data
        self.image_augmentation_info = image_augmentation_info
        # cals
        self.images, self.labels = self.images_data.images, self.images_data.labels
        self.aug_types, self.aug_prop = self.image_augmentation_info.aug_types, self.image_augmentation_info.aug_prop

    def rotate_image(self):
        # Sample images to rotate
        num_images = len(self.images)
        num_to_rotate = int(num_images * self.aug_prop)
        # image_indices = random.sample(range(num_images), num_to_rotate)
        image_indices = random.choices(range(num_images), k=num_to_rotate)
        # Apply rotation to sampled images
        # rotated_images = np.zeros_like(self.images[:num_to_rotate])
        first_image = self.images[0]
        rotated_images = np.tile(first_image, reps=(num_to_rotate, 1, 1, 1))
        for i, index in enumerate(image_indices):
            image = self.images[index]
            label = self.labels[index]
            angle = random.randint(-180, 180)
            rows, cols, _ = image.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            rotated = cv2.warpAffine(image, M, (cols, rows))
            rotated_images[i] = rotated

        # Concatenate rotated images with original images
        self.images = np.concatenate((self.images, rotated_images[:len(image_indices)]), axis=0)
        self.labels = np.concatenate((self.labels, self.labels[image_indices]), axis=0)
        # Check that the number of images and labels is the same
        if len(self.images) != len(self.labels):
            raise ValueError(f"Number of images and labels must be the same, {len(self.images), len(self.labels)}")

    def zoom_image(self):
        # zoom in factor 1-1.5 randomly selected
        # Retrieve size and number of channels of the first image in the data
        first_image = self.images[0]
        orig_height, orig_width, orig_channels = first_image.shape

        # Sample images to zoom
        num_images = len(self.images)
        num_to_zoom = int(num_images * self.aug_prop)
        image_indices = random.choices(range(num_images), k=num_to_zoom)

        # Apply zooming to sampled images
        zoomed_images = np.zeros_like(self.images[:num_to_zoom])
        for i, index in enumerate(image_indices):
            image = self.images[index]
            label = self.labels[index]
            height, width, channels = image.shape
            scale = random.uniform(1.0, min(1.5, float(orig_height) / height, float(orig_width) / width))
            new_height = int(scale * height)
            new_width = int(scale * width)
            # Resize the image while maintaining the aspect ratio using bilinear interpolation
            zoomed = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            # Resize the zoomed image to the size of the first image in the data
            zoomed = cv2.resize(zoomed, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)
            # Pad the image to maintain the original shape
            pad_top = (orig_height - new_height) // 2
            pad_bottom = orig_height - new_height - pad_top
            pad_left = (orig_width - new_width) // 2
            pad_right = orig_width - new_width - pad_left
            # Clamp the padding values to zero and the size of the padded image
            pad_top = max(0, pad_top)
            pad_bottom = max(0, pad_bottom)
            pad_left = max(0, pad_left)
            pad_right = max(0, pad_right)
            pad_top = min(orig_height, pad_top)
            pad_bottom = min(orig_height, pad_bottom)
            pad_left = min(orig_width, pad_left)
            pad_right = min(orig_width, pad_right)
            # Create a new zero-filled array with the original size and number of channels
            padded = np.zeros((orig_height, orig_width, orig_channels), dtype=np.uint8)
            # Copy the zoomed image to the center of the new array
            padded[pad_top:pad_top + new_height, pad_left:pad_left + new_width, :channels] = zoomed
            zoomed_images[i] = padded

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
                                 validation_prop=self.images_data.validation_prop)
        self.images_data = images_data
        return images_data
