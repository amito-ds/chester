import numpy as np
import torch
import torch.hub as hub
import torch.nn as nn
from IPython import get_ipython
from torch import optim
from torchvision import models
from tqdm import tqdm

from diamond.user_classes import ImagesData, ImageModel


class ImageModelTraining:
    def __init__(self, images_data: ImagesData,
                 image_model: ImageModel):
        self.images_data = images_data
        self.image_model = image_model
        self.num_classes = len(np.unique(images_data.labels))
        self.num_epochs = image_model.num_epochs
        self.train_loader, self.val_loader = images_data.create_data_loaders(
            batch_size=self.image_model.batch_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training using {self.device}")
        if 'google.colab' in str(get_ipython()) and self.device == "cpu":
            print(
                f"If you are using Google Colab, you may want to change the runtime to use a GPU or TPU by "
                f"selecting 'Runtime' -> 'Change runtime type'.")
        self.evaluate_predictions = None

    def get_model(self):
        supported_models = ["EfficientNetB0", "EfficientNetB4", "EfficientNetB7", "ResNet50", "ResNet101",
                            "DenseNet121", "DenseNet161", "DenseNet201"]
        if self.image_model.network_name == "EfficientNetB0":
            model = hub.load('rwightman/pytorch-image-models', 'efficientnet_b0', pretrained=True)
        elif self.image_model.network_name == "EfficientNetB4":
            model = hub.load('rwightman/pytorch-image-models', 'efficientnet_b4', pretrained=True)
        elif self.image_model.network_name == "EfficientNetB7":
            model = hub.load('rwightman/pytorch-image-models', 'efficientnet_b7', pretrained=True)
        elif self.image_model.network_name == "ResNet50":
            model = models.resnet50(pretrained=True)
        elif self.image_model.network_name == "ResNet101":
            model = models.resnet101(pretrained=True)
        elif self.image_model.network_name == "DenseNet121":
            model = models.densenet121(pretrained=True)
        elif self.image_model.network_name == "DenseNet161":
            model = models.densenet161(pretrained=True)
        elif self.image_model.network_name == "DenseNet169":
            model = models.densenet169(pretrained=True)
        elif self.image_model.network_name == "DenseNet201":
            model = models.densenet201(pretrained=True)
        else:
            raise ValueError(
                f"Unsupported network name: {self.image_model.network_name}, please choose one of: {supported_models}")
        return model

    def load_model(self):
        model = self.get_model().to(self.device)

        # remove the specified number of layers from the top
        if self.image_model.remove_last_layers_num > 0:
            print(f"Removing last {self.image_model.remove_last_layers_num} layers for {self.image_model.network_name}")
            if "resnet" in self.image_model.network_name.lower():
                num_features = model.fc.in_features
                model.fc = nn.Sequential(
                    nn.Dropout(p=self.image_model.dropout),
                    nn.Linear(num_features, self.num_classes)
                )
            elif "inception" in self.image_model.network_name.lower():
                num_features = model.fc.in_features
                model.fc = nn.Identity()
                model.fc_new = nn.Linear(num_features, self.num_classes)
            else:
                for i in range(self.image_model.remove_last_layers_num):
                    try:
                        num_features = model.classifier.in_features
                    except:
                        num_features = model.classifier[0].in_features

                    model.classifier = nn.Sequential(
                        *list(model.classifier.children())[:-1],
                        nn.Dropout(p=self.image_model.dropout),
                        nn.Linear(num_features, self.num_classes)
                    )

        # make the remaining layers non-trainable
        for param in model.parameters():
            param.requires_grad = True

        return model

    def train_model(self, model, criterion, optimizer):
        train_loss, evaluate_accuracy = 0, 0
        for epoch in tqdm(range(self.num_epochs)):
            train_loss = self.train_model_epoch(model, self.train_loader, criterion, optimizer)
            evaluate_accuracy = self.evaluate_model(model=model, data_loader=self.val_loader, device=self.device)
            tqdm.write(
                f"Epoch {epoch + 1}/{self.num_epochs} - Train Loss: {train_loss:.4f} - Evaluate Accuracy: {evaluate_accuracy:.4f}")
        return train_loss, evaluate_accuracy

    def train_model_epoch(self, model, dataloader, criterion, optimizer):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader, 0):
            optimizer.zero_grad()

            inputs, labels = inputs.to(self.device).float(), labels.to(self.device)

            try:
                model = model.to(self.device)
                outputs = model(inputs).to(self.device)
            except Exception as e:
                print(f"Error while running model: {e}")
                print("Input size:", inputs.size())

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)

        return epoch_loss

    @staticmethod
    def evaluate_model(model, data_loader, device):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in data_loader:
                images, labels = data
                images, labels = images.to(device).float(), labels.to(device)

                outputs = model(images).to(device)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return accuracy

    @staticmethod
    def get_eval_prediction(model, data_loader, device, num_classes):
        # Get the predictions for the eval set
        eval_preds = []
        with torch.no_grad():
            for data in data_loader:
                images, labels = data
                images, labels = images.to(device).float(), labels.to(device)
                outputs = model(images).to(device)
                for i in range(outputs.size(0)):  # iterate over batch size
                    output = outputs[i]
                    _, predicted = torch.max(output, 0)
                    eval_preds.append(predicted)
        eval_preds = torch.stack(eval_preds)
        return eval_preds

    def run(self):
        model = self.load_model()
        criterion = nn.CrossEntropyLoss()  # future: more loss functions, for bb, regression
        optimizer_params = self.image_model.optimizer_params
        print("\nTraining Specifications: ", self.image_model.network_parameters)
        optimizer = optim.Adam(model.parameters(), **optimizer_params)  # get rid of lr
        train_loss, val_loss = self.train_model(model, criterion, optimizer)  # get rid of epochs
        self.evaluate_predictions = self.get_eval_prediction(
            model, self.val_loader, num_classes=self.num_classes, device=self.device)
        return model, train_loss, val_loss, self.evaluate_predictions
