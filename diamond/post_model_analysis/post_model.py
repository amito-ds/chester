from diamond.model_training.model_results import ImageModelResults
from diamond.user_classes import ImagesData, ImagePostModelSpec
import pandas as pd
import torch


class ImagePostModelAnalysis:
    def __init__(self, model_list: list[ImageModelResults],
                 images_data: ImagesData,
                 image_post_model: ImagePostModelSpec,
                 plot=True,
                 diamond_collector=None):
        self.model_list = model_list
        self.images_data = images_data
        self.image_post_model = image_post_model
        self.plot = plot
        self.best_model = self.model_list[0].model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.diamond_collector = {} if diamond_collector is None else diamond_collector
        self.network_parameters = self.model_list[0].network_parameters
        self.image_model_training = self.model_list[0].image_model_training
        self.train_loader, self.val_loader = self.images_data.create_data_loaders(
            batch_size=self.network_parameters["_batch_size"])
        self.prediction = self.get_prediction()

    def get_prediction(self):
        predicted = self.image_model_training.evaluate_predictions
        self.diamond_collector["eval predictions"] = predicted
        return predicted

    def compare_models(self):
        data = []
        for model in self.model_list:
            data.append({**model.network_parameters,
                         "train_loss": model.train_loss, "val_loss": model.val_loss})
        df = pd.DataFrame(data)
        print("Comparing All Models: ")
        self.diamond_collector["models comparison"] = df
        return df

    def confusion_matrix(self, plot):
        if "class" not in self.images_data.problem_type:
            return None
        # update in diamond_collector
        if plot:
            pass
        pass

    def precision_recall(self, plot):
        if "class" not in self.images_data.problem_type:
            return None
        # self.predictions VS self.images_data.labels_val
        # calc the confusion matrix
        if plot:
            # plot with appropriare title
            pass
        pass

    def run(self):
        if self.image_post_model.is_compare_models:
            self.compare_models()
        if self.image_post_model.is_confusion_matrix:
            self.confusion_matrix(plot=self.plot)
        if self.image_post_model.is_precision_recall:
            self.precision_recall(plot=self.plot)
