import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from chester.feature_stats.utils import create_pretty_table
from chester.model_monitor import calculate_scores_utils
from chester.model_training.data_preparation import CVData
from chester.model_training.models.chester_models.catboost.catboost_utils import calculate_catboost_metrics_scores
from chester.zero_break.problem_specification import DataInfo

boostrap_message = "The following plots display the bootstrap distribution of the metrics, " \
                   "\nUsing a violin plot and histogram: " \
                   "\n\tℹ️ The violin plot shows the median, " \
                   "quartiles, and other important summary statistics of the data, " \
                   "\n\tℹ️ While the histogram provides a visual representation of the " \
                   "frequency of values in the data. " \
                   "\nThese plots give you a sense of how the metrics are distributed " \
                   "\nand can be used to identify any patterns or outliers in the data."


class ModelBootstrap:
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
        self.metrics = self.get_metrics_functions()
        from chester.util import ReportCollector, REPORT_PATH
        self.rc = ReportCollector(REPORT_PATH)

    def get_metrics_functions(self):
        metric_functions = []
        metrics = self.data_info.metrics_detector_val
        for metric in metrics:
            metric_function = self.get_metric_function(metric)
            if metric_function is not None:
                metric_functions.append(metric_function)
        return metric_functions

    def get_metric_function(self, metric):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
        from sklearn.metrics import roc_auc_score

        if metric == 'Accuracy':
            return accuracy_score
        elif metric == 'Precision':
            return precision_score
        elif metric == 'Recall':
            return recall_score
        elif metric == 'F1':
            return f1_score
        elif metric == "MSE":
            return mean_squared_error
        elif metric == "MAE":
            return mean_absolute_error
        elif metric == "MAPE":
            return mean_absolute_percentage_error
        elif metric == 'ROC':
            return roc_auc_score
        else:
            return None

    def bootstrap_metrics(self, B=200, sample=200):
        n = len(self.y_test)
        bootstrap_metrics = []
        for i in range(B):
            sample_indexes = np.random.randint(0, n, sample)
            X_sample = self.X_test.iloc[sample_indexes]
            y_sample = self.y_test.iloc[sample_indexes]
            y_pred = self.model.predict(X_sample)

            try:
                bootstrap_metrics.append(
                    calculate_catboost_metrics_scores(
                        y_sample, prediction=y_pred, metrics_list=self.metrics,
                        problem_type=self.data_info.problem_type_val)
                )
            except:
                bootstrap_metrics.append(
                    calculate_scores_utils.calculate_metrics_scores(
                        y=y_sample, prediction=y_pred, metrics_list=self.metrics,
                        problem_type=self.data_info.problem_type_val)
                )
        return bootstrap_metrics

    def plot(self):
        print(boostrap_message)
        import seaborn as sns
        metrics = pd.DataFrame(self.bootstrap_metrics())
        self.rc.save_object(create_pretty_table(metrics.sample(20)), "\nBoostrap metrics, sample 20 results:")
        for metric_name in metrics.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            sns.violinplot(x=metrics[metric_name], ax=ax1)
            ax1.set_title(metric_name)
            sns.histplot(x=metrics[metric_name], ax=ax2)
            plt.show()
            plt.close()
