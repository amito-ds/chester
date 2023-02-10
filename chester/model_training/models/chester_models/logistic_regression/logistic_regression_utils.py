import random
from typing import List

import numpy as np
import pandas as pd

from chester.model_training.data_preparation import CVData
from chester.model_training.data_preparation import Parameter
from chester.model_training.models.chester_models.base_model_utils import calculate_metrics_scores
from chester.model_training.models.chester_models.base_model_utils import is_metric_higher_is_better
from chester.model_training.models.chester_models.hp_generator import HPGenerator
from chester.model_training.models.chester_models.logistic_regression.logistic_regression_model import \
    LogisticRegressionModel
from chester.zero_break.problem_specification import DataInfo


def train_logistic_regression(X_train, y_train, parameters: list, data_info: DataInfo):
    """
    Trains a baseline model using the given parameters.
    :param X_train: The training data features (unused in this function)
    :param y_train: The training data labels
    :param parameters: parameters list
    :return: A trained baseline model
    """
    model = LogisticRegressionModel(parameters, data_info)
    model.fit(X_train, y_train)
    return model


def predict_logistic_regression(model, X):
    """
    Makes predictions using a baseline model
    :param model: trained baseline model
    :param X: The data to make predictions on (unused in this function)
    :return: A list of predictions
    """
    return model.transform(X)


def logistic_regression_with_outputs(cv_data: CVData,
                                     metrics: list,
                                     target_col: str,
                                     parameters: list,
                                     data_info: DataInfo,
                                     problem_type=None,
                                     ):
    results = []
    for i, (X_train, y_train, X_test, y_test) in enumerate(cv_data.format_splits()):
        model = train_logistic_regression(X_train, y_train, parameters=parameters, data_info=data_info)
        prediction = predict_logistic_regression(model, X_test)
        prediction_train = predict_logistic_regression(model, X_train)
        scores = calculate_metrics_scores(y_test, prediction, metrics, problem_type)
        results.append({'type': 'test', 'fold': i, **scores})
        scores = calculate_metrics_scores(y_train, prediction_train, metrics, problem_type)
        results.append({'type': 'train', 'fold': i, **scores})
    return results, model


def compare_models(results):
    all_results = [(pd.DataFrame(result), model) for result, model in results]
    # print("all_results", all_results[0][0])
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


default_parameters = {
    'penalty': 'l2',
    'C': 1.0,
    'solver': 'lbfgs',
    'max_iter': 100
}

logistic_regression_default_parameters = [
    Parameter('penalty', default_parameters['penalty']),
    Parameter('C', default_parameters['C']),
    Parameter('solver', default_parameters['solver']),
    Parameter('max_iter', default_parameters['max_iter'])
]


def generate_logistic_regression_configs(k: int = 10, best_practice_prob=0.33) -> List[List[Parameter]]:
    hp_generator = HPGeneratorLR(n_models=k, best_practice_prob=best_practice_prob)
    parameter_format = hp_generator.hp_format(hp_generator.generate_configs())
    return parameter_format


class HPGeneratorLR(HPGenerator):
    def __init__(self, best_practice_configs: list = None,
                 categorical_sample_configs: list = None,
                 n_models=9,
                 best_practice_prob=0.33):
        super().__init__(best_practice_configs, categorical_sample_configs, n_models, best_practice_prob)
        self.best_practice_configs = self.load_best_practice_configs()

    @staticmethod
    def load_best_practice_configs():
        return [
            {'penalty': 'l1', 'C': 0.1, 'solver': 'saga'},
            {'penalty': 'l2', 'C': 0.1, 'solver': 'newton-cg'},
            {'penalty': 'elasticnet', 'C': 0.1, 'solver': 'saga', 'l1_ratio': 0.5},
            {'penalty': 'l2', 'C': 0.01, 'solver': 'lbfgs'},
            {'penalty': 'l1', 'C': 0.5, 'solver': 'saga'},
            {'penalty': 'l2', 'C': 0.5, 'solver': 'newton-cg'},
            {'penalty': 'elasticnet', 'C': 0.5, 'solver': 'saga', 'l1_ratio': 0.5},
            {'penalty': 'l2', 'C': 0.001, 'solver': 'lbfgs'},
            {'penalty': 'l1', 'C': 1, 'solver': 'saga'},
            {'penalty': 'l2', 'C': 1, 'solver': 'newton-cg'},
            {'penalty': 'elasticnet', 'C': 1, 'solver': 'saga', 'l1_ratio': 0.5},
        ]

    def generate_random_config(self) -> dict:
        cat_configs = [
            {'penalty': 'l1', 'solver': 'saga'},
            {'penalty': 'l2', 'solver': 'newton-cg'},
            {'penalty': 'elasticnet', 'solver': 'saga', 'l1_ratio': 0.5},
            {'penalty': 'l2', 'solver': 'lbfgs'},
            {'penalty': 'l1', 'solver': 'saga'},
            {'penalty': 'l2', 'solver': 'newton-cg'},
            {'penalty': 'elasticnet', 'solver': 'saga', 'l1_ratio': 0.5},
            {'penalty': 'l2', 'solver': 'lbfgs'},
            {'penalty': 'l1', 'solver': 'saga'},
            {'penalty': 'l2', 'solver': 'newton-cg'},
            {'penalty': 'elasticnet', 'solver': 'saga', 'l1_ratio': 0.5},
        ]

        cat_config = random.choice(cat_configs)
        config = cat_config.copy()
        config['C'] = self.generate_c_value()
        if 'l1_ratio' in cat_config:
            config['l1_ratio'] = self.generate_l1_ratio_random_configs()

        return config

    @staticmethod
    def generate_l1_ratio_random_configs() -> float:
        l1_ratio_values = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
        return random.choice(l1_ratio_values)

    @staticmethod
    def generate_c_value():
        choice = random.choice([0, 1])
        if choice == 0:
            return np.random.uniform(0.0001, 100)
        else:
            return np.random.uniform(np.log10(1.001), np.log10(100))
