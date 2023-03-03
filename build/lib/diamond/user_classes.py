import random
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


class ImagesData:
    def __init__(self, images, labels, validation_prop, image_shape):
        self.images = images
        self.labels = labels
        self.validation_prop = validation_prop
        self.image_shape = image_shape
        self.problem_type = self.get_problem_type()
        # reshape
        self.images = self.images.reshape((-1,) + self.image_shape)
        self.is_colored = True if len(self.image_shape) > 2 else False
        if self.is_colored:
            self.images_to_show = np.transpose(self.images, (0, 2, 3, 1))
        else:
            self.images_to_show = np.transpose(self.images, (0, 1, 2))
        self.images_train, self.labels_train, self.images_val, self.labels_val = self.split()

    def split(self):
        assert 0 <= self.validation_prop < 0.8, "validation proportion should be in range (0, 0.8)"
        num_val = int(self.validation_prop * len(self.images))
        indices = np.random.permutation(len(self.images))
        val_indices, train_indices = indices[:num_val], indices[num_val:]

        images_train, labels_train = self.images[train_indices], self.labels[train_indices]
        images_val, labels_val = self.images[val_indices], self.labels[val_indices]

        return images_train, labels_train, images_val, labels_val

    def create_data_loaders(self, batch_size):
        train_dataset = TensorDataset(torch.from_numpy(self.images_train), torch.from_numpy(self.labels_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(torch.from_numpy(self.images_val), torch.from_numpy(self.labels_val))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # add tqdm progress bar to train_loader
        train_loader = tqdm(train_loader, total=len(train_loader))

        return train_loader, val_loader

    def get_splits(self):
        return self.images_train, self.labels_train, self.images_val, self.labels_val

    @staticmethod
    def get_problem_type():
        return "classification"

    def plot_images(self):
        num_images = len(self.images_to_show)
        if num_images > 100:
            image_indices = range(num_images - 100, num_images)
        else:
            image_indices = range(num_images)

        num_cols = int(math.floor(math.sqrt(len(image_indices))))
        num_rows = int(math.ceil(len(image_indices) / num_cols))

        fig, ax = plt.subplots(num_rows, num_cols, figsize=(12, 12))
        for i, index in enumerate(image_indices):
            row = i // num_cols
            col = i % num_cols
            ax[row, col].imshow(self.images_to_show[index])
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


class ImagePostModelSpec:
    def __init__(self, plot,
                 is_compare_models=True,
                 is_confusion_matrix=True,
                 is_precision_recall=True):
        self.plot = plot
        self.is_compare_models = is_compare_models
        self.is_confusion_matrix = is_confusion_matrix
        self.is_precision_recall = is_precision_recall
