import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from IPython import get_ipython
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score

from diamond.model_training.model_results import ImageModelResults
from diamond.user_classes import ImagesData, ImagePostModelSpec


class ImagePostModelAnalysis:
    def __init__(self, model_list,
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
        self.label_dict = self.images_data.label_dict

        self.diamond_collector = {} if diamond_collector is None else diamond_collector
        self.network_parameters = self.model_list[0].network_parameters
        self.image_model_training = self.model_list[0].image_model_training
        self.train_loader, self.val_loader = self.images_data.create_data_loaders(
            batch_size=self.network_parameters["_batch_size"])
        self.eval_predictions = self.get_prediction()
        self.eval_labels = self.images_data.labels_val

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
        code = """diamond_collector["models comparison"]"""
        print(f"Comparing for all Models is Saved, for exploration, run {code}")
        self.diamond_collector["models comparison"] = df
        return df

    @staticmethod
    def replace_predictions(eval_labels_unique, eval_predictions):
        updated_predictions = []
        for eval_prediction in eval_predictions:
            if eval_prediction not in eval_labels_unique:
                updated_predictions.append(-1)
            else:
                updated_predictions.append(eval_prediction)
        return updated_predictions

    def map_labels(self, x):
        for k, v in self.label_dict.items():
            if v == x:
                return k
        return x  # Return the original value if it is not in the label_dict dictionary

    def confusion_matrix(self, plot):
        if "class" not in self.images_data.problem_type:
            return None
        vmap_labels = np.vectorize(self.map_labels)
        eval_labels = vmap_labels(self.eval_labels)
        if torch.cuda.is_available():
            eval_predictions = vmap_labels(self.eval_predictions.cpu().numpy())
        else:
            eval_predictions = vmap_labels(self.eval_predictions)

        cm = confusion_matrix(eval_labels, eval_predictions)

        # Get the actual class names in sorted order
        sorted_class_names = sorted(self.label_dict, key=lambda x: self.label_dict[x])

        # Create a heatmap plot with class names as tick labels
        if plot:
            fig, ax = plt.subplots(figsize=(8, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, ax=ax, cbar=False,
                        xticklabels=sorted_class_names, yticklabels=sorted_class_names)
            ax.set_xlabel('Predicted label')
            ax.set_ylabel('True label')
            ax.set_title('Confusion Matrix')
            plt.show()

        self.diamond_collector["confusion matrix"] = cm
        return cm

    def precision_recall(self):
        import warnings
        warnings.filterwarnings("ignore", message="invalid value encountered in divide")
        if "class" not in self.images_data.problem_type:
            return None

        vmap_labels = np.vectorize(self.map_labels)
        eval_labels = vmap_labels(self.eval_labels)
        if torch.cuda.is_available():
            eval_predictions = vmap_labels(self.eval_predictions.cpu().numpy())
        else:
            eval_predictions = vmap_labels(self.eval_predictions)

        if torch.cuda.is_available():
            eval_pred = eval_predictions.cpu().numpy()
            precision = precision_score(eval_labels, eval_pred, average=None)
            recall = recall_score(eval_labels, eval_pred, average=None)
            accuracy = accuracy_score(eval_labels, eval_pred)
        else:
            precision = precision_score(eval_labels, eval_predictions, average=None)
            recall = recall_score(eval_labels, eval_predictions, average=None)
            accuracy = accuracy_score(eval_labels, eval_predictions)
        # Calculate precision and recall for each class
        f1_score = 2 * (precision * recall) / (precision + recall)

        # Create a dataframe with the results
        label_values = np.unique(eval_labels)
        results = pd.DataFrame({"Class": label_values,
                                "Accuracy": accuracy,
                                "Precision": precision,
                                "Recall": recall,
                                "F1-Score": f1_score})

        # Display the results
        self.diamond_collector["Precision Recall"] = results
        return results

    def run(self):
        if self.image_post_model.is_compare_models:
            self.compare_models()
        if self.image_post_model.is_confusion_matrix:
            self.confusion_matrix(plot=self.plot)
        if self.image_post_model.is_precision_recall:
            try:
                self.precision_recall()
            except:
                pass
