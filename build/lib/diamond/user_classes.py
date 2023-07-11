import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


class ImagesData:
    def __init__(self, images, labels, validation_prop, image_shape, label_dict, for_model_training=True):
        self.images = images
        self.raw_images = self.images
        self.raw_image = None
        self.image_shape = image_shape
        # print("image shape origin", self.image_shape)
        self.is_colored = True if len(self.image_shape) > 2 else False

        self.format_images()  # format images
        self.labels = labels
        self.validation_prop = validation_prop
        self.label_dict = label_dict
        self.for_model_training = for_model_training
        self.validate()  # validation
        self.problem_type = self.get_problem_type()
        self.label_hanlder()
        self.images_to_show = None
        self.image_to_show()
        if self.for_model_training:
            self.images_train, self.labels_train, self.images_val, self.labels_val = self.split()
        if self.is_colored:
            print("Colored image")
        else:
            print("Grayscale image")

    def format_images(self):
        # print("Formatting images of type", type(self.images))
        np_images = []
        image_list = []

        if isinstance(self.images, pd.DataFrame):
            # Convert DataFrame to numpy array
            np_images = self.images.to_numpy()

            # Convert each numpy array to PIL Image object and add to list
            for img_arr in np_images:
                img = Image.fromarray(img_arr)
                image_list.append(img)

        elif isinstance(self.images, list):
            # Loop through each image and convert to numpy array and PIL Image object
            for idx, image in enumerate(self.images):
                if isinstance(image, str):
                    # if image is a file path, open it using PIL and convert to numpy array
                    with Image.open(image) as img:
                        img_array = np.array(img)
                elif isinstance(image, np.ndarray):
                    # if image is a numpy array, use it directly
                    img_array = image
                else:
                    try:
                        img_array = np.array(image)
                    except:
                        img_array = image

                # Append PIL Image object and numpy array to lists
                image_list.append(Image.fromarray(img_array))
                np_images.append(img_array)

        elif isinstance(self.images, np.ndarray):
            # Convert numpy array to PIL Image object
            for img_arr in self.images:
                if self.is_colored:
                    mode = 'RGB'
                    height, width = sorted(self.image_shape, reverse=True)[:2]
                    img_arr = img_arr.reshape((height, width, 3))
                else:
                    mode = 'L'  # L stands for grayscale mode
                    img_arr = img_arr.reshape(self.image_shape)
                img = Image.fromarray(img_arr, mode=mode)
                image_list.append(img)
                np_images.append(img_arr)
        else:
            raise TypeError("Unsupported image format:", type(self.images))

        # Update the instance variables with the formatted image data
        self.images = np.array(np_images)
        self.raw_image = image_list[0]
        self.raw_images = image_list

    def validate(self):
        if self.labels is None:
            pass
        assert len(self.images) == len(self.labels), "# of Images must be == # of labels"

    def image_to_show(self):
        try:
            images = self.images.reshape((-1,) + self.image_shape)
            if self.is_colored:
                self.images_to_show = np.transpose(images, (0, 2, 3, 1))
            else:
                self.images_to_show = np.transpose(images, (0, 1, 2))
        except:
            pass

    def label_hanlder(self):
        if isinstance(self.labels, pd.DataFrame):
            self.labels = self.labels.to_numpy()

    def split(self):
        assert 0 <= self.validation_prop < 0.5, "validation proportion should be in range (0, 0.5)"
        print("Before splitting, these are the shapes")
        print("images:", self.images.shape)

        num_images = len(self.images)
        num_val_images = int(num_images * self.validation_prop)
        num_train_images = num_images - num_val_images

        # Shuffle the images and labels before splitting
        shuffled_indices = np.random.permutation(num_images)
        shuffled_images = np.array(self.images)[shuffled_indices]
        shuffled_labels = self.labels[shuffled_indices]

        # Split the images and labels into training and validation sets
        train_images = shuffled_images[:num_train_images]
        train_labels = shuffled_labels[:num_train_images]
        val_images = shuffled_images[num_train_images:]
        val_labels = shuffled_labels[num_train_images:]

        # Convert to correct shape
        train_images = train_images.reshape((-1,) + self.image_shape)
        val_images = val_images.reshape((-1,) + self.image_shape)

        print("After splitting, these are the shapes")
        print("train_images:", train_images.shape)
        print("val_images:", val_images.shape)

        return train_images, train_labels, val_images, val_labels

    def create_data_loaders(self, batch_size):
        labels_train = np.array(self.labels_train)
        train_dataset = TensorDataset(torch.from_numpy(self.images_train), torch.from_numpy(labels_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        labels_val = np.array(self.labels_val)
        val_dataset = TensorDataset(torch.from_numpy(self.images_val), torch.from_numpy(labels_val))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # add tqdm progress bar to train_loader
        train_loader = tqdm(train_loader, total=len(train_loader))

        return train_loader, val_loader

    def get_splits(self):
        return self.images_train, self.labels_train, self.images_val, self.labels_val

    @staticmethod
    def get_problem_type():
        return "classification"

    # def plot_images(self, plot_sample):
    #     print("Total Images:", len(self.images))
    #     if self.images_to_show is None:
    #         self.images_to_show = self.images
    #     images = self.images_to_show
    #     num_images = len(images)
    #     plot_sample = min(plot_sample, num_images)
    #     if num_images > 100:
    #         image_indices = range(num_images - plot_sample + 1, num_images)
    #     else:
    #         image_indices = random.sample(range(num_images), plot_sample)
    #
    #     num_cols = int(math.floor(math.sqrt(len(image_indices))))
    #     num_rows = int(math.ceil(len(image_indices) / num_cols))
    #
    #     fig, ax = plt.subplots(num_rows, num_cols, figsize=(12, 12))
    #     for i, index in enumerate(image_indices):
    #         row = i // num_cols
    #         col = i % num_cols
    #         ax[row, col].imshow(images[index])
    #         ax[row, col].axis('off')
    #         if self.labels is not None:
    #             ax[row, col].set_title(str(self.labels[index]))
    #     plt.show()

    def plot_images(self, plot_sample):
        print("Total Images:", len(self.images))
        if self.images_to_show is None:
            self.images_to_show = self.images
        images = self.images_to_show
        num_images = len(images)
        plot_sample = min(plot_sample, num_images)
        if num_images > 100:
            image_indices = range(num_images - plot_sample + 1, num_images)
        else:
            image_indices = random.sample(range(num_images), plot_sample)

        if len(image_indices) == 1:
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.imshow(images[image_indices[0]])
            ax.axis('off')
            if self.labels is not None:
                ax.set_title(str(self.labels[image_indices[0]]))
        else:
            num_cols = int(math.floor(math.sqrt(len(image_indices))))
            num_rows = int(math.ceil(len(image_indices) / num_cols))
            fig, ax = plt.subplots(num_rows, num_cols, figsize=(12, 12))
            for i, index in enumerate(image_indices):
                row = i // num_cols
                col = i % num_cols
                ax[row, col].imshow(images[index])
                ax[row, col].axis('off')
                if self.labels is not None:
                    ax[row, col].set_title(str(self.labels[index]))
        plt.show()


class ImagesAugmentationInfo:

    def __init__(self, aug_types=None, aug_prop=0.05):
        self.aug_types = aug_types
        self.aug_prop = aug_prop
        # calc
        if self.aug_types is None:
            self.aug_types = ["zoom", "rotate"]


class ImageModel:
    def __init__(self,
                 network_name="EfficientNetB0",
                 remove_num_last_layers=1,
                 batch_size=64,
                 num_epochs=3,
                 optimizer_params=None,
                 dropout=0.5
                 ):
        self.network_name = network_name
        self.remove_last_layers_num = remove_num_last_layers
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.optimizer_params = optimizer_params
        self.dropout = dropout
        if self.optimizer_params is None:
            self.optimizer_params = {"lr": 0.001, 'weight_decay': 0.0001}
        self.network_parameters = self.optimizer_params.copy()
        self.network_parameters["_model_name"] = self.network_name
        self.network_parameters["_num_epochs"] = self.num_epochs
        self.network_parameters["_batch_size"] = self.batch_size
        self.network_parameters["_dropout"] = self.dropout
        self.network_parameters["_remove_last_layers_num layers"] = self.remove_last_layers_num


class ImageModels:
    def __init__(self, image_model_list=None):
        self.image_model_list = image_model_list
        if image_model_list is None:
            self.image_model_list = [ImageModel()]
        self.n_models = len(self.image_model_list)


class ImagePostModelSpec:
    def __init__(self, plot,
                 is_compare_models=True,
                 is_confusion_matrix=True,
                 is_precision_recall=True):
        self.plot = plot
        self.is_compare_models = is_compare_models
        self.is_confusion_matrix = is_confusion_matrix
        self.is_precision_recall = is_precision_recall


class ImageDescriptionSpec:
    def __init__(self, max_length=16, num_beams=4):
        self.max_length = max_length
        self.num_beams = num_beams
        self.gen_kwargs = {"max_length": self.max_length, "num_beams": self.num_beams}
