import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

from chester.feature_stats.utils import create_pretty_table
from chester.model_analyzer.model_analysis import AnalyzeMessages
from chester.model_training.data_preparation import CVData
from chester.post_model_analysis.post_regression import VisualizeRegressionResults
from chester.zero_break.problem_specification import DataInfo


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
        self.model.retrain(self.X_train, self.y_train)
        self.predict_test = self.model.predict(self.X_test)
        from chester.util import ReportCollector, REPORT_PATH
        self.rc = ReportCollector(REPORT_PATH)

    def plot_feature_importance(self, X_train: pd.DataFrame, top_feat=30):
        feature_importance = self.model.model.feature_importances_
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        feature_names = X_train.columns
        important_idx = np.argsort(feature_importance)
        data = {'feature_names': feature_names[important_idx],
                'feature_importance': feature_importance[important_idx]}
        df = pd.DataFrame(data)
        print(AnalyzeMessages().feature_importance_message())
        df.sort_values(by='feature_importance', ascending=False, inplace=True)
        plt.figure(figsize=(15, 13))
        plt.title(f'Feature Importance (top {top_feat} features)')

        feature_importance = {}
        # Iterate over the rows of the DataFrame and add each feature and its importance to the dictionary
        for i, row in df[0:100].iterrows():
            feature = row['feature_names']
            importance = int(round(row['feature_importance']))
            feature_importance[feature] = importance
        self.rc.save_object(feature_importance,
                            text="Top 100 Feature importance (in %, 100 = the most important one):")
        sns.barplot(x='feature_importance', y='feature_names', data=df[0:top_feat])
        self.plot_wordcloud_importance(df=df)
        plt.show()

    @staticmethod
    def plot_wordcloud_importance(df, title="Feature Importance Cloud"):
        """
        Plot word cloud of features weighted by their importance.
        :param features_importance: List of tuples (feature name, feature importance).
        :param title: Title of the plot.
        :return: None.
        """
        df = df.sort_values(by='feature_importance', ascending=False)
        features_importance = list(zip(df['feature_names'], df['feature_importance']))

        wordcloud = WordCloud(
            random_state=21,
            normalize_plurals=True,
            max_words=100).generate_from_frequencies(dict(features_importance))
        plt.figure(figsize=(12, 8))
        plt.title(title, fontsize=15)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.show()
        plt.close()

    def shap_values(self, X_train: pd.DataFrame, shap):
        explainer = shap.Explainer(self.model, X_train, check_additivity=False)
        shap_values = explainer(X_train, check_additivity=False)
        plt.title("SHAP values for train set")
        print(AnalyzeMessages().shap_values_message())
        shap.summary_plot(shap_values, X_train)

    def analyze(self,
                shap_values: bool = True,
                coefficients: bool = True,
                confusion_matrix: bool = True,
                roc_curve: bool = True,
                learning_curve: bool = True,
                feature_importance: bool = True,
                regression_visual: bool = True) -> None:
        if 'regression' in self.data_info.problem_type_val.lower():
            if regression_visual:
                VisualizeRegressionResults(self.y_test, self.predict_test).all_plots()
        if feature_importance:
            try:
                self.plot_feature_importance(self.X_train)
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
            if 'class' in self.data_info.problem_type_val.lower():
                try:
                    self.confusion_matrix(self.X_test, self.y_test)
                except:
                    pass
        if learning_curve:
            self.learning_curve(self.X_train, self.y_train)
        if roc_curve:
            try:
                self.roc_curve(self.X_test, self.y_test)
            except:
                try:
                    self.roc_curve_multiclass(self.X_test, self.y_test)
                except:
                    return None

    def confusion_matrix(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        print(AnalyzeMessages().confusion_matrix_message())

        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        self.rc.save_object(cm, text="confusion matrix on the test set")

        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        plt.close()

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
            plt.close()
        except:
            plt.close()
            return None

    def roc_curve_multiclass(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        from sklearn.metrics import roc_auc_score, roc_curve
        from sklearn.preprocessing import label_binarize
        from scipy import interp

        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        y_pred = self.model.predict_proba(X_test)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(y_test_bin.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred[:, i])
            roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_pred[:, i])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(y_test_bin.shape[1])]))

        mean_tpr = np.zeros_like(all_fpr)
        for i in range(y_test_bin.shape[1]):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        mean_tpr /= y_test_bin.shape[1]
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = roc_auc_score(y_test_bin, y_pred, average='macro')

        plt.figure()
        plt.plot(fpr["macro"], tpr["macro"], color='darkorange', lw=1, label='Macro-average ROC curve (area = {0:0.2f})'
                                                                             ''.format(roc_auc["macro"]))

        for i in range(y_test_bin.shape[1]):
            plt.plot(fpr[i], tpr[i], lw=1, label='ROC curve of class {0} (area = {1:0.2f})'
                                                 ''.format(i, roc_auc[i]))

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

    def learning_curve(self, X: pd.DataFrame, y: pd.Series) -> None:
        import os
        import sys

        original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        try:
            from sklearn.model_selection import learning_curve
            metric_name = self.data_info.metrics_detector_val[0].lower()
            metric_name = 'roc_auc' if metric_name == 'roc' else metric_name
            metric_name = 'neg_mean_squared_error' if metric_name == 'mse' else metric_name
            train_sizes, train_scores, test_scores = learning_curve(self.model.model, X, y, cv=5, scoring=metric_name)
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)

            # create the pandas DataFrame
            learning_curve_df = pd.DataFrame({
                'train_sizes': train_sizes,
                'train_scores_mean': train_scores_mean,
                'train_scores_std': train_scores_std,
                'test_scores_mean': test_scores_mean,
                'test_scores_std': test_scores_std
            })
            self.rc.save_object(obj=create_pretty_table(learning_curve_df),
                                text="Learning curve by training examples")

            if len(test_scores_std) == 0:
                plt.close()
                return None
            plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
            plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
            plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                             alpha=0.1, color="r")
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
        finally:
            plt.close()
            sys.stderr = original_stderr

    def coefficients(self) -> None:
        try:
            coef = self.model.coef_[0]
            abs_coef = np.abs(self.model.coef_[0])
            top_100_coef_idx = np.argsort(abs_coef)[::-1][:100]
            top_100_coef_values = [self.model.coef_[0][i] for i in top_100_coef_idx]
            self.rc.save_object(top_100_coef_values, text="First 100 model coefficients: ")
            feature_names = self.model.feature_names
            if len(coef) < 10:
                plt.bar(np.arange(len(coef)), coef)
                plt.xticks(np.arange(len(coef)), feature_names, rotation=90)
                plt.title("Coefficients for logistic regression model")
                plt.xlabel("Features")
                plt.ylabel("Values")
            else:
                sns.violinplot(coef, inner="stick")
                plt.title("Coefficients distribution for logistic regression model")
                plt.xlabel("Coefficients")
                print(AnalyzeMessages().coefficients_message())
            plt.show()
            plt.close()

            coef_dict = {}
            for i in range(len(feature_names)):
                coef_dict[feature_names[i]] = abs(coef[i])
            sorted_coef_dict = dict(sorted(coef_dict.items(), key=lambda x: x[1], reverse=True)[:100])
            self.rc.save_object(obj=sorted_coef_dict,
                                text="Top 100 feature with highest abs(coef): ")
        except:
            plt.close()
            return None
