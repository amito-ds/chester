import torch.nn as nn
from torch import optim
from torchvision.models import efficientnet_b0

from diamond.user_classes import ImagesData, ImageModel
import torch.hub as hub

import torch


class ImageModelTraining:
    def __init__(self, images_data: ImagesData,
                 image_model: ImageModel):
        self.images_data = images_data
        self.image_model = image_model
        self.num_epochs = image_model.num_epochs
        self.train_loader, self.val_loader = images_data.create_data_loaders(
            batch_size=self.image_model.batch_size)  # train_loader, val_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_model(self):
        if self.image_model.network_name == "EfficientNetB0":
            model = hub.load('rwightman/pytorch-image-models', 'efficientnet_b0', pretrained=True)
        elif self.image_model.network_name == "EfficientNetB4":
            model = hub.load('rwightman/pytorch-image-models', 'efficientnet_b4', pretrained=True)
        elif self.image_model.network_name == "EfficientNetB7":
            model = hub.load('rwightman/pytorch-image-models', 'efficientnet_b7', pretrained=True)
        elif self.image_model.network_name == "ResNet50":
            model = hub.load('pytorch/vision', 'resnet50', pretrained=True)
        elif self.image_model.network_name == "ResNet101":
            model = hub.load('pytorch/vision', 'resnet101', pretrained=True)
        elif self.image_model.network_name == "DenseNet121":
            model = hub.load('pytorch/vision', 'densenet121', pretrained=True)
        elif self.image_model.network_name == "VGG16":
            model = hub.load('pytorch/vision', 'vgg16', pretrained=True)
        elif self.image_model.network_name == "InceptionV3":
            model = hub.load('pytorch/vision', 'inception_v3', pretrained=True)
        else:
            raise ValueError(f"Unsupported network name: {self.image_model.network_name}")
        # remove the specified number of layers from the top
        if self.image_model.remove_last_layers_num > 0:
            for i in range(self.image_model.remove_last_layers_num):
                model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

        # make the remaining layers non-trainable
        for param in model.parameters():
            param.requires_grad = True

        return model

    def train_model(self, model, criterion, optimizer):
        train_loss, val_loss = 0, 0
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            train_loss = self.train_model_epoch(model, self.train_loader, criterion, optimizer)
            val_loss = self.evaluate_model(model, self.val_loader)
            print(f"Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f}")
        return train_loss, val_loss

    def train_model_epoch(self, model, dataloader, criterion, optimizer):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader, 0):
            optimizer.zero_grad()

            inputs, labels = inputs.to(self.device), labels.to(self.device)

            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)

        return epoch_loss

    def evaluate_model(self, model, data_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in data_loader:
                images, labels = data
                outputs = model(images.float())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return accuracy

    def run(self):
        model = self.load_model()
        criterion = nn.CrossEntropyLoss()  # future: more loss functions, for bb, regression
        optimizer_params = self.image_model.optimizer_params
        print("Training Specifications: ", self.image_model.network_parameters)
        optimizer = optim.Adam(model.parameters(), **optimizer_params)  # get rid of lr
        train_loss, val_loss = self.train_model(model, criterion, optimizer)  # get rid of epochs
        return model, train_loss, val_loss
