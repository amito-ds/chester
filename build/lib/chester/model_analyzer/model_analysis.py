import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn import metrics

from chester.model_training.data_preparation import CVData


class ModelAnalyzer:
    def __init__(self, model):
        self.model = model

    def shap_values(self, X_train: pd.DataFrame):
        import shap
        explainer = shap.Explainer(self.model, X_train, check_additivity=False)
        shap_values = explainer(X_train, check_additivity=False)
        plt.title("SHAP values for train set")
        print(AnalyzeMessages().shap_values_message())
        shap.summary_plot(shap_values, X_train)

    def coefficients(self, X_train: pd.DataFrame) -> None:
        coef = self.model.coef_[0]
        feature_names = X_train.columns
        if len(coef) < 10:
            plt.bar(np.arange(len(coef)), coef)
            plt.xticks(np.arange(len(coef)), feature_names, rotation=90)
            plt.title("Coefficients for Logistic Regression Model")
            plt.xlabel("Features")
            plt.ylabel("Values")
            plt.show()
            plt.close()
        else:
            sns.violinplot(coef, inner="stick")
            plt.title("Coefficients distribution for logistic regression model")
            plt.xlabel("Coefficients")
            print(AnalyzeMessages().coefficients_message())
        plt.show()
        plt.close()

    def analyze(self, X_train: pd.DataFrame, y_train: pd.Series,
                X_test: pd.DataFrame, y_test: pd.Series,
                model, plot_shap_values: bool = True, coefficients: bool = True,
                confusion_matrix: bool = True,
                roc_curve: bool = True, learning_curve: bool = True,
                feature_importance: bool = True) -> None:
        unique_classes = len(np.unique(y_train))
        if feature_importance:
            try:
                self.plot_feature_importance(X_train)
            except:
                try:
                    self.plot_simple_feature_importance(X_train)
                except:
                    pass
        if plot_shap_values:
            if unique_classes == 2:
                try:
                    self.shap_values(X_train)
                except:
                    pass
        if coefficients:
            try:
                self.coefficients(X_train)
            except:
                pass
        if confusion_matrix:
            try:
                self.confusion_matrix(X_test, y_test)
            except:
                pass
        if roc_curve:
            if unique_classes == 2:
                self.roc_curve(X_test, y_test)
        if learning_curve:
            self.learning_curve(X_train, y_train)

    def confusion_matrix(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        print(AnalyzeMessages().confusion_matrix_message())

        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        plt.close()

    def roc_curve(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        from sklearn.metrics import roc_curve, auc
        try:
            y_pred = self.model.predict_proba(X_test)
            fpr, tpr, thresholds = roc_curve(y_test, y_pred[:, 1])
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
            print(AnalyzeMessages().roc_curve_message())
            plt.show()
            plt.close()
        except:
            return None

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
            print(AnalyzeMessages().learning_curve_message())
            plt.show()
            plt.close()
        except:
            plt.close()
            return None

    def plot_simple_feature_importance(self, X_train: pd.DataFrame):
        feature_importance = self.model.feature_importances_
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        feature_names = X_train.columns
        important_idx = np.argsort(feature_importance)
        data = {'feature_names': feature_names[important_idx], 'feature_importance': feature_importance[important_idx]}
        df = pd.DataFrame(data)
        sns.barplot(y='feature_names', x='feature_importance', data=df)
        plt.xlabel("Feature importance")
        plt.show()
        plt.close()

    def plot_feature_importance(self, X_train: pd.DataFrame):
        feature_importance = self.model.feature_importances_
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        feature_names = X_train.columns
        important_idx = np.argsort(feature_importance)
        data = {'feature_names': feature_names[important_idx], 'feature_importance': feature_importance[important_idx]}
        df = pd.DataFrame(data)
        print(AnalyzeMessages().feature_importance_message())
        fig = px.bar(df, x='feature_importance', y='feature_names', orientation='h', text='feature_importance')
        fig.show()
        plt.close()


class AnalyzeMessages:
    def shap_values_message(self):
        return "SHAP values can be used to understand the importance of each feature in the model's" \
               " predictions.\n The plot shows the average absolute SHAP value of each feature for all " \
               "the samples in the test set.\n Features with higher absolute SHAP values have a greater " \
               "impact on the model's predictions.\n"

    def coefficients_message(self):
        return "The coefficient plot shows the weight of each feature in the model's predictions.\n" \
               " Positive coefficients indicate that the feature has a positive relationship with" \
               " the target variable\n, while negative coefficients indicate a negative relationship.\n" \
               " The size of the boxplot represents the range of values for each feature.\n"

    def performance_metrics_message(self):
        return "The performance metrics table shows the results of different evaluation metrics" \
               " applied on the model's predictions\n. These metrics can give you an idea of how well" \
               " the model is performing in terms of accuracy, precision, recall, and other measures.\n"

    def confusion_matrix_message(self):
        return "The confusion matrix shows the number of true positive, " \
               "true negative, false positive, and false negative predictions made by the model.\n " \
               "This can give you an idea of how well the model is able to distinguish between the " \
               "different classes.\n"

    def roc_curve_message(self):
        return "The ROC curve shows the trade-off between true positive rate (sensitivity) " \
               "and false positive rate (1-specificity) for different threshold settings.\n T" \
               "he AUC (Area under the curve) value gives an overall measure of the model's performance.\n"

    def learning_curve_message(self):
        return "The learning curve displays model performance as training samples increase.\n" \
               "High training and low validation suggest overfitting,\n" \
               "low training and high validation suggest underfitting.\n"

    def feature_importance_message(self):
        return "The feature importance plot shows the relative importance of each feature in the model's predictions.\n" \
               " Features with higher importance have a greater impact on the model's predictions and are more useful\n" \
               " for making accurate predictions.\n"


def analyze_model(model, cv_data: CVData, target_label='target'):
    X_train, X_test, y_train, y_test = \
        cv_data.train_data.drop(columns=[target_label]), \
            cv_data.test_data.drop(columns=[target_label]), \
            cv_data.train_data[target_label], \
            cv_data.test_data[target_label]
    analyzer = ModelAnalyzer(model)
    analyzer.analyze(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                     model=model)


def get_default_metrics(y):
    n_classes = len(np.unique(y))
    if n_classes == 2:
        return [metrics.accuracy_score, metrics.f1_score, metrics.mean_squared_error, metrics.r2_score]
    elif n_classes > 2:
        return [metrics.accuracy_score, metrics.f1_score]
    else:
        return [metrics.mean_squared_error, metrics.r2_score]


import Levenshtein


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
        return 'grey'
    if value < thresholds[closest_metric][0]:
        return 'red'
    elif value < thresholds[closest_metric][1]:
        return 'yellow'
    elif value < thresholds[closest_metric][2]:
        return 'green'
    else:
        return 'bright green'
