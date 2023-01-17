import difflib
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

from mdoel_training.data_preparation import CVData
import Levenshtein
#
# You need to pass the trained model to the shap.Explainer() function, not just the name of the model.
#
# The shap.summary_plot() function can take the shap values and the data as input, but it's also possible to pass a specific feature to the shap.summary_plot() function, for example shap.summary_plot(shap_values, X_train, feature_names='feature_name')
#
# The shap.summary_plot() function can also take other parameters such as plot_type which can be set to "bar" to show shap values as a bar chart, class_names to show the names of the classes in a classification problem and color to set the color of the shap values.
class ModelAnalyzer:
    def __init__(self, model):
        self.model = model

    def shap_values(self, X_train: pd.DataFrame):
        ## TO DO handle model types, get it as an argument
        explainer = shap.Explainer(self.model, X_train)
        shap_values = explainer(X_train)
        plt.title("SHAP values for train set")
        shap.summary_plot(shap_values, X_train)
        # try:
        #     explainer = shap.Explainer(self.model, X_train, check_additivity=False)
        #     shap_values = explainer(X_train, check_additivity=False)
        #     plt.title("SHAP values for train set")
        #     shap.summary_plot(shap_values, X_train)
        # except:
        #     pass

    def coefficients(self) -> None:
        print("print coef")
        coef = self.model.coef_[0]
        if len(coef) < 50:
            plt.bar(np.arange(len(coef)), coef)
            plt.title("Coefficients for logistic regression model")
            plt.xlabel("Features")
            plt.ylabel("Values")
        else:
            sns.violinplot(coef, inner="stick")
            plt.title("Coefficients distribution for logistic regression model")
            plt.xlabel("Coefficients")
        plt.show()

    def performance_metrics(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,
                            metric_functions: List) -> None:
        try:
            train_metrics = {metric.__name__: metric(y_train, self.model.predict(X_train)) for metric in
                             metric_functions}
            test_metrics = {metric.__name__: metric(y_test, self.model.predict(X_test)) for metric in metric_functions}
        except ValueError as e:
            if len(np.unique(y_train)) == 2:
                # converting categorical labels to 0,1
                encoder = LabelEncoder()
                y_train = encoder.fit_transform(y_train)
                y_test = encoder.transform(y_test)
                train_predictions = encoder.transform(self.model.predict(X_train))
                test_predictions = encoder.transform(self.model.predict(X_test))
                train_metrics = {metric.__name__: metric(y_train, train_predictions) for metric in metric_functions}
                test_metrics = {metric.__name__: metric(y_test, test_predictions) for metric in metric_functions}
            else:
                print(f"Error: {e}. Skipping performance metrics calculation.")
        train_metrics = pd.DataFrame(train_metrics.items(), columns=["Metric Name", "Value"])
        test_metrics = pd.DataFrame(test_metrics.items(), columns=["Metric Name", "Value"])
        print("Train set performance:")
        print(train_metrics)
        print("Test set performance:")
        print(test_metrics)
        # population_pyramid_plot(train_metrics, test_metrics)

    def confusion_matrix(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    def roc_curve(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        from sklearn.metrics import roc_curve, auc
        try:
            y_pred = self.model.predict_proba(X_test)
            y_test_binary = (y_test == 'pirates').astype(int)
            fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic')
            plt.legend(loc="lower right")
            plt.show()
        except:
            pass

    def learning_curve(self, X: pd.DataFrame, y: pd.Series) -> None:
        try:
            from sklearn.model_selection import learning_curve
            train_sizes, train_scores, test_scores = learning_curve(self.model, X, y, cv=5, scoring='accuracy')
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
            plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1, color="r")
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1, color="g")
            plt.grid()
            plt.xlabel("Training examples")
            plt.ylabel("Accuracy Score")
            plt.title("Learning Curve")
            plt.legend(loc="best")
            plt.show()
        except:
            pass

    import matplotlib.pyplot as plt

    def plot_feature_importance(self, model, X_train):
        if type(model).__name__ == 'LGBMClassifier':
            feature_importance = model.feature_importances_
            feature_importance = 100.0 * (feature_importance / feature_importance.max())
            feature_names = X_train.columns
            important_idx = np.where(feature_importance)[0]
            important_features = feature_names[important_idx]
            plt.barh(range(important_idx.shape[0]), feature_importance[important_idx], align='center')
            plt.yticks(range(important_idx.shape[0]), important_features)
            plt.xlabel("Feature importance")
            plt.ylabel("Feature")
            plt.show()
        elif type(model).__name__ == 'RandomForestClassifier':
            feature_importance = model.feature_importances_
            feature_names = X_train.columns
            important_idx = np.where(feature_importance)[0]
            important_features = feature_names[important_idx]
            plt.barh(range(important_idx.shape[0]), feature_importance[important_idx], align='center')
            plt.yticks(range(important_idx.shape[0]), important_features)
            plt.xlabel("Feature importance")
            plt.ylabel("Feature")
            plt.show()
        else:
            print(f'Feature importance not available for {type(model).__name__} model')

    def analyze(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,
                metric_functions: List, model, shap_values: bool = True, coefficients: bool = False,
                performance_metrics: bool = True, confusion_matrix: bool = True,
                roc_curve: bool = True, learning_curve: bool = True, feature_importance: bool = True) -> None:

        messages = AnalyzeMessages()
        if feature_importance:
            # print(messages.feature_importance_message())
            self.plot_feature_importance(model, X_train)
        if shap_values:
            print(messages.shap_values_message())
            print(" shap X_train shape", X_train.shape)
            self.shap_values(X_train)
        if coefficients:
            print(messages.coefficients_message())
            self.coefficients()
        if performance_metrics:
            print(messages.performance_metrics_message())
            self.performance_metrics(X_train, y_train, X_test, y_test, metric_functions)
        if confusion_matrix:
            print(messages.confusion_matrix_message())
            self.confusion_matrix(X_test, y_test)
        if roc_curve:
            print(messages.roc_curve_message())
            self.roc_curve(X_test, y_test)
        if learning_curve:
            print(messages.learning_curve_message())
            self.learning_curve(X_train, y_train)


class AnalyzeMessages:
    def shap_values_message(self):
        return "SHAP values can be used to understand the importance of each feature in the model's" \
               " predictions. The plot shows the average absolute SHAP value of each feature for all " \
               "the samples in the test set. Features with higher absolute SHAP values have a greater " \
               "impact on the model's predictions."

    def coefficients_message(self):
        return "The coefficient plot shows the weight of each feature in the model's predictions." \
               " Positive coefficients indicate that the feature has a positive relationship with" \
               " the target variable, while negative coefficients indicate a negative relationship." \
               " The size of the boxplot represents the range of values for each feature."

    def performance_metrics_message(self):
        return "The performance metrics table shows the results of different evaluation metrics" \
               " applied on the model's predictions. These metrics can give you an idea of how well" \
               " the model is performing in terms of accuracy, precision, recall, and other measures."

    def confusion_matrix_message(self):
        return "The confusion matrix shows the number of true positive, " \
               "true negative, false positive, and false negative predictions made by the model. " \
               "This can give you an idea of how well the model is able to distinguish between the " \
               "different classes."

    def roc_curve_message(self):
        return "The ROC curve shows the trade-off between true positive rate (sensitivity) " \
               "and false positive rate (1-specificity) for different threshold settings. T" \
               "he AUC (Area under the curve) value gives an overall measure of the model's performance."

    def learning_curve_message(self):
        return "The learning curve shows the model's performance as the number of " \
               "training samples increases. A high training score and a low validation " \
               "score indicates overfitting, while a low training and high validation score " \
               "indicates underfitting."


def analyze_model(model, cv_data: CVData, target_label='target'):
    X_train, X_test, y_train, y_test = \
        cv_data.train_data.drop(columns=[target_label]), \
            cv_data.test_data.drop(columns=[target_label]), \
            cv_data.train_data[target_label], \
            cv_data.test_data[target_label]
    metric_functions = get_default_metrics(y_train)
    analyzer = ModelAnalyzer(model)
    analyzer.analyze(X_train, y_train, X_test, y_test, metric_functions, model)


def get_default_metrics(y):
    n_classes = len(np.unique(y))
    if n_classes == 2:
        return [metrics.accuracy_score, metrics.f1_score, metrics.mean_squared_error, metrics.r2_score]
    elif n_classes > 2:
        return [metrics.accuracy_score, metrics.f1_score]
    else:
        return [metrics.mean_squared_error, metrics.r2_score]


import jellyfish


def get_traffic_light(metric_name, value):
    thresholds = {"Accuracy": (0.8, 0.9, 0.95),
                  "F1-Score": (0.7, 0.8, 0.9),
                  "Precision": (0.7, 0.8, 0.9),
                  "Recall": (0.7, 0.8, 0.9),
                  "ROC-AUC": (0.7, 0.8, 0.9),
                  "BLEU Score": (0.7, 0.8, 0.9),
                  "METEOR Score": (0.7, 0.8, 0.9),
                  "Rouge Score": (0.7, 0.8, 0.9),
                  "CIDEr Score": (0.7, 0.8, 0.9),
                  "Embedding Average Cosine Similarity": (0.7, 0.8, 0.9)}
    metric_name = metric_name.lower()
    closest_metric = None
    closest_distance = float('inf')
    for metric in thresholds.keys():
        distance = Levenshtein.distance(metric_name, metric.lower())
        if distance < closest_distance:
            closest_metric = metric
            closest_distance = distance
    if closest_metric is None:
        return 'w'
    if value < thresholds[closest_metric][0]:
        return 'r'
    elif value < thresholds[closest_metric][1]:
        return (1, 0.5, 0)
    elif value < thresholds[closest_metric][2]:
        return 'y'
    else:
        return 'g'


import matplotlib.pyplot as plt
import seaborn as sns

import seaborn as sns
import matplotlib.pyplot as plt


def population_pyramid_plot(train_metrics, test_metrics=None):
    # Get the colors for each metric
    train_metrics['Color'] = train_metrics.apply(lambda x: get_traffic_light(x['Metric Name'], x['Value']), axis=1)
    if test_metrics is not None:
        test_metrics['Color'] = test_metrics.apply(lambda x: get_traffic_light(x['Metric Name'], x['Value']), axis=1)

    # Set up the plot
    fig, ax = plt.subplots()
    ax.set_ylim(0, 100)
    ax.set_xlim(-1, 1)
    ax.axis('off')

    def plot_metrics(metrics, ax):
        x = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
        y = np.array([0, metrics['Value'], 100, 100, metrics['Value'], 0, 0, metrics['Value'], 100])
        # ax.fill_betweenx(y, x, color=metrics['Color'])
        ax.text(0, metrics['Value'].iloc[0] / 2, str(metrics['Value'].iloc[0]), ha='center', va='center')
        # ax.text(0, metrics['Value'] / 2, str(metrics['Value']), ha='center', va='center')

    plot_metrics(train_metrics, ax)
    if test_metrics is not None:
        plot_metrics(test_metrics, ax)
    # Add labels and show the plot
    plt.title("Population Pyramid Metrics")
    plt.xlabel("Metric Name")
    plt.ylabel("Value")
    plt.show()
