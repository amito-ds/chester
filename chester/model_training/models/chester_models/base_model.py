from chester.model_training.data_preparation import CVData
from chester.zero_break.problem_specification import DataInfo


class BaseModel:
    def __init__(self, data_info: DataInfo, cv_data: CVData, num_models_to_compare=10, best_practice_prop=0.33):
        self.data_info = data_info
        self.cv_data = cv_data
        self.num_models_to_compare = num_models_to_compare
        self.best_practice_prop = best_practice_prop
        self.best_model = None
        self.best_metrics = {}
        try:
            self.feature_names = self.cv_data.train_data.columns
        except:
            pass

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

    def get_metrics_functions(self):
        metric_functions = []
        metrics = self.data_info.metrics_detector_val
        for metric in metrics:
            metric_function = self.get_metric_function(metric)
            if metric_function is not None:
                metric_functions.append(metric_function)
        return metric_functions

    def conf_generator(self):
        pass

    def train_single_model(self, params: list):
        pass

    def get_best_model(self):
        pass
