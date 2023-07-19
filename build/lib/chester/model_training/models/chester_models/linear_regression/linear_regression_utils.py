import random
from typing import List

import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning

from chester.model_training.data_preparation import CVData
from chester.model_training.data_preparation import Parameter
from chester.model_training.models.chester_models.base_model_utils import is_metric_higher_is_better
from chester.model_training.models.chester_models.hp_generator import HPGenerator
from chester.model_training.models.chester_models.linear_regression.linear_regression import LinearRegressionModel
from chester.zero_break.problem_specification import DataInfo


def train_linear_regression(X_train, y_train, parameters: list, data_info: DataInfo):
    """
    Trains a baseline model using the given parameters.
    :param X_train: The training data features (unused in this function)
    :param y_train: The training data labels
    :param parameters: parameters list
    :return: A trained baseline model
    """
    model = LinearRegressionModel(parameters, data_info)
    model.fit(X_train, y_train)
    return model


def predict_linear_regression(model, X):
    """
    Makes predictions using a baseline model
    :param model: trained baseline model
    :param X: The data to make predictions on (unused in this function)
    :return: A list of predictions
    """
    return model.transform(X)


def linear_regression_with_outputs(cv_data: CVData,
                                   metrics: list,
                                   target_col: str,
                                   parameters: list,
                                   data_info: DataInfo,
                                   ):
    results = []
    for i, (X_train, y_train, X_test, y_test) in enumerate(cv_data.format_splits()):
        model = train_linear_regression(X_train, y_train, parameters=parameters, data_info=data_info)
        prediction = predict_linear_regression(model, X_test)
        prediction_train = predict_linear_regression(model, X_train)
        scores = calculate_linear_regression_metrics_scores(y_test, prediction, metrics, data_info.problem_type_val)
        results.append({'type': 'test', 'fold': i, **scores})
        scores = calculate_linear_regression_metrics_scores(y_train, prediction_train, metrics,
                                                            data_info.problem_type_val)
        results.append({'type': 'train', 'fold': i, **scores})
    return results, model


def compare_models(results):
    all_results = [(pd.DataFrame(result), model) for result, model in results]
    metric_name = [col for col in all_results[0][0].columns if col not in ['type', 'fold']][0]
    sort_ascending = is_metric_higher_is_better(metric_name)
    best_result = None
    best_model = None
    best_value = None
    for (result, model) in all_results:
        test_result = result[result['type'] == 'test'].groupby('fold').mean(numeric_only=True).reset_index()
        mean_value = test_result[metric_name].mean()
        if best_value is None or \
                (sort_ascending and mean_value > best_value) \
                or (not sort_ascending and mean_value < best_value):
            best_value = mean_value
            best_result = result
            best_model = model
    return best_result, best_model


default_parameters = {}


def generate_linear_regression_configs(k: int = 10, best_practice_prob=0.33) -> List[List[Parameter]]:
    hp_generator = HPGeneratoEN(n_models=k, best_practice_prob=best_practice_prob)
    parameter_format = hp_generator.hp_format(hp_generator.generate_configs())
    return parameter_format


class HPGeneratoEN(HPGenerator):
    def __init__(self, best_practice_configs: list = None,
                 categorical_sample_configs: list = None,
                 n_models=9,
                 best_practice_prob=0.33):
        super().__init__(best_practice_configs, categorical_sample_configs, n_models, best_practice_prob)
        self.best_practice_configs = self.load_best_practice_configs()

    @staticmethod
    def load_best_practice_configs():
        return [
            {**default_parameters, 'alpha': 0, 'l1_ratio': 0},  # linear
            {**default_parameters, 'alpha': 0.1, 'l1_ratio': 0},  # ridge
            {**default_parameters, 'alpha': 0.1, 'l1_ratio': 1},  # lasso
            {**default_parameters, 'alpha': 0.1, 'l1_ratio': 0.1},
            {**default_parameters, 'alpha': 0.1, 'l1_ratio': 0.5},
            {**default_parameters, 'alpha': 0.1, 'l1_ratio': 0.7},
            {**default_parameters, 'alpha': 0.1, 'l1_ratio': 0.9},
            {**default_parameters, 'alpha': 0.1, 'l1_ratio': 1.0},
            {**default_parameters, 'alpha': 0.5, 'l1_ratio': 0.1},
            {**default_parameters, 'alpha': 0.5, 'l1_ratio': 0.5},
            {**default_parameters, 'alpha': 0.5, 'l1_ratio': 0.7},
            {**default_parameters, 'alpha': 0.5, 'l1_ratio': 0.9},
            {**default_parameters, 'alpha': 1, 'l1_ratio': 0},  # ridge
            {**default_parameters, 'alpha': 1, 'l1_ratio': 1},  # lasso
            {**default_parameters, 'alpha': 0.5, 'l1_ratio': 1.0},
            {**default_parameters, 'alpha': 1.0, 'l1_ratio': 0.1},
            {**default_parameters, 'alpha': 1.0, 'l1_ratio': 0.5},
            {**default_parameters, 'alpha': 1.0, 'l1_ratio': 0.7},
            {**default_parameters, 'alpha': 1.0, 'l1_ratio': 0.9}
        ]

    def generate_random_config(self) -> dict:
        method = random.choices(['Elastic net', 'Lasso', 'Ridge'], weights=[0.5, 0.25, 0.25])[0]
        if method == 'Elastic net':
            alpha = random.uniform(0, 1)
            l1_ratio = random.uniform(0, 1)
        elif method == 'Lasso':
            alpha = random.uniform(0, 1)
            l1_ratio = 1
        elif method == 'Ridge':
            alpha = random.uniform(0, 1)
            l1_ratio = 0
        return {**default_parameters, 'alpha': alpha, 'l1_ratio': l1_ratio}


def calculate_linear_regression_metric_score(y_true, y_pred, metric, problem_type_val):
    import warnings
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    metric_name = metric.__name__
    try:
        return metric_name, metric(y_true, y_pred)
    except:
        try:
            if problem_type_val in ["Binary regression", "Binary classification"]:
                return metric_name, metric(y_true, y_pred, average='binary')
            elif problem_type_val == "Multiclass classification":
                return metric_name, metric(y_true, y_pred, average='macro')
        except:
            pass
    return None, None


def calculate_linear_regression_metrics_scores(y, prediction, metrics_list, problem_type=None):
    results = {}
    for metric in metrics_list:
        metric_name, metric_value = calculate_linear_regression_metric_score(y, prediction, metric, problem_type)
        results[metric_name] = metric_value
    return results
