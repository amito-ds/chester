import numpy as np
import pandas as pd
import plotly.express as px
from chester.model_analyzer.model_analysis import AnalyzeMessages
from chester.model_training.data_preparation import CVData
from chester.zero_break.problem_specification import DataInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn import metrics

from chester.model_training.data_preparation import CVData
from chester.model_training.models.scoring import calculate_score_model


class PostModelAnalysis:
    def __init__(self, cv_data: CVData, data_info: DataInfo, model):
        self.cv_data = cv_data
        self.data_info = data_info
        self.model = model
        self.X_train = self.cv_data.train_data.drop(columns=[self.cv_data.target_column])
        self.y_train = self.cv_data.train_data[self.cv_data.target_column]
        self.X_test = self.cv_data.test_data.drop(columns=[self.cv_data.target_column])
        self.y_test = self.cv_data.test_data[self.cv_data.target_column]
        # retrain the model
        print("retraining model...")
        self.model.retrain(self.X_train, self.y_train)

    def plot_feature_importance(self, X_train: pd.DataFrame, top_feat: int = 25):
        feature_importance = self.model.model.feature_importances_
        total_feat = len(feature_importance)
        feature_importance = feature_importance[0:min(top_feat, total_feat)]
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        feature_names = X_train.columns
        important_idx = np.argsort(feature_importance)
        data = {'feature_names': feature_names[important_idx], 'feature_importance': feature_importance[important_idx]}
        df = pd.DataFrame(data)
        # print(AnalyzeMessages().feature_importance_message())
        fig = px.bar(df, x='feature_importance', y='feature_names', orientation='h', text='feature_importance')
        fig.show()

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

    def shap_values(self, X_train: pd.DataFrame, shap):
        explainer = shap.Explainer(self.model, X_train, check_additivity=False)
        shap_values = explainer(X_train, check_additivity=False)
        plt.title("SHAP values for train set")
        print(AnalyzeMessages().shap_values_message())
        shap.summary_plot(shap_values, X_train)

    def analyze(self,
                shap_values: bool = True,
                coefficients: bool = True,
                confusion_matrix: bool = False,
                roc_curve: bool = True,
                learning_curve: bool = True,
                feature_importance: bool = True) -> None:
        if feature_importance:
            try:
                self.plot_feature_importance(self.X_train)
            except:
                try:
                    self.plot_simple_feature_importance(self.X_train)
                except:
                    pass
        if shap_values:
            try:
                import shap as shp
                self.shap_values(self.X_train, shp)
            except:
                pass
        if coefficients:
            try:
                self.coefficients()
            except:
                pass
        if confusion_matrix:
            try:
                # try only if len(y) < k and len(predictions) < k
                self.confusion_matrix(self.X_test, self.y_test)
            except:
                pass
        if roc_curve:
            try:
                self.roc_curve(self.X_test, self.y_test)
            except:
                pass
        if learning_curve:
            try:
                self.learning_curve(self.X_train, self.y_train)
            except:
                pass

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

    def roc_curve(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        from sklearn.metrics import roc_curve, auc
        try:
            y_pred = self.model.predict(X_test)
            fpr, tpr, thresholds = roc_curve(y_test, y_pred)
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
        except:
            pass

    def learning_curve(self, X: pd.DataFrame, y: pd.Series) -> None:
        try:
            from sklearn.model_selection import learning_curve
            train_sizes, train_scores, test_scores = learning_curve(self.model.model, X, y, cv=5, scoring='accuracy')
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
        except:
            pass

    def coefficients(self) -> None:
        coef = self.model.coef_[0]
        if len(coef) < 10:
            plt.bar(np.arange(len(coef)), coef)
            plt.title("Coefficients for logistic regression model")
            plt.xlabel("Features")
            plt.ylabel("Values")
        else:
            sns.violinplot(coef, inner="stick")
            plt.title("Coefficients distribution for logistic regression model")
            plt.xlabel("Coefficients")
            print(AnalyzeMessages().coefficients_message())
        plt.show()
