from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import shap

from mdoel_training.data_preparation import CVData
import seaborn as sns


class ModelAnalyzer:
    def __init__(self, model):
        self.model = model

    def shap_values(self, X_test: pd.DataFrame) -> None:
        explainer = shap.Explainer(self.model, X_test)
        shap_values = explainer(X_test)
        plt.title("SHAP values for train set")
        shap.summary_plot(shap_values, X_test)

    def coefficients(self) -> None:
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
        train_metrics = {metric.__name__: metric(y_train, self.model.predict(X_train)) for metric in metric_functions}
        test_metrics = {metric.__name__: metric(y_test, self.model.predict(X_test)) for metric in metric_functions}
        train_metrics = pd.DataFrame(train_metrics.items(), columns=["Metric Name", "Value"])
        test_metrics = pd.DataFrame(test_metrics.items(), columns=["Metric Name", "Value"])
        print("Train set performance:")
        print(train_metrics)
        print("Test set performance:")
        print(test_metrics)

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

    def learning_curve(self, X: pd.DataFrame, y: pd.Series) -> None:
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

    def analyze(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,
                metric_functions: List, shap_values: bool = True, coefficients: bool = True,
                performance_metrics: bool = True, confusion_matrix: bool = True,
                roc_curve: bool = True, learning_curve: bool = True) -> None:

        messages = AnalyzeMessages()
        if shap_values:
            self.shap_values(X_test)
            print(messages.shap_values_message())
        if coefficients:
            self.coefficients()
            print(messages.coefficients_message())
        if performance_metrics:
            self.performance_metrics(X_train, y_train, X_test, y_test, metric_functions)
            print(messages.performance_metrics_message())
        if confusion_matrix:
            self.confusion_matrix(X_test, y_test)
            print(messages.confusion_matrix_message())
        if roc_curve:
            self.roc_curve(X_test, y_test)
            print(messages.roc_curve_message())
        if learning_curve:
            self.learning_curve(X_train, y_train)
            print(messages.learning_curve_message())


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


def analyze_model(model, cv_data: CVData, target_label: str, metric_functions=None):
    if metric_functions is None:
        metric_functions = []
    analyzer = ModelAnalyzer(model)
    X_train = cv_data.train_data.drop(columns=[target_label])
    y_train = cv_data.train_data[target_label]
    X_test = cv_data.test_data.drop(columns=[target_label])
    y_test = cv_data.test_data[target_label]
    analyzer.analyze(X_train, y_train, X_test, y_test, metric_functions)
