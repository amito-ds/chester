from collections import defaultdict
from typing import List

import pandas as pd

from chester.model_training.data_preparation import CVData
from chester.model_training.data_preparation import Parameter
from chester.model_training.models.chester_models.base_model_utils import calculate_metrics_scores
from chester.model_training.models.chester_models.base_model_utils import is_metric_higher_is_better
from chester.model_training.models.chester_models.logistic_regression.logistic_regression_model import \
    LogisticRegressionModel


def train_logistic_regression(X_train, y_train, parameters: list):
    """
    Trains a baseline model using the given parameters.
    :param X_train: The training data features (unused in this function)
    :param y_train: The training data labels
    :param parameters: parameters list
    :return: A trained baseline model
    """
    model = LogisticRegressionModel(parameters)
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
                                     problem_type=None,
                                     ):
    results = []
    for i, (X_train, y_train, X_test, y_test) in enumerate(cv_data.format_splits()):
        model = train_logistic_regression(X_train, y_train, parameters=parameters)
        prediction = predict_logistic_regression(model, X_test)
        prediction_train = predict_logistic_regression(model, X_train)
        # print("prediction_train", prediction_train)
        print("metrics")
        # print(metrics)
        scores = calculate_metrics_scores(y_test, prediction, metrics, problem_type)
        print(scores)
        results.append({'type': 'test', 'fold': i, **scores})
        scores = calculate_metrics_scores(y_train, prediction_train, metrics, problem_type)
        results.append({'type': 'train', 'fold': i, **scores})
    return results, model


def compare_models(results):
    all_results = [(pd.DataFrame(result), model) for result, model in results]
    print("all_results", all_results[0][0])
    metric_name = [col for col in all_results[0][0].columns if col not in ['type', 'fold']][0]
    sort_ascending = is_metric_higher_is_better(metric_name)
    best_result = None
    best_model = None
    best_value = None
    for (result, model) in all_results:
        test_result = result[result['type'] == 'test'].groupby('fold').mean().reset_index()
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


def generate_logistic_regression_configs(k: int) -> List[List[Parameter]]:
    # List of additional configurations to test
    additional_confs = [
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
    ]
    # List to store the final configurations
    logistic_regression_parameters = []
    for conf in additional_confs[:k]:
        # Create a dictionary to store the final configuration
        final_conf = defaultdict(lambda: None, default_parameters)
        final_conf.update(conf)
        # Convert the dictionary to a list of Parameter objects
        final_conf = [Parameter(key, value) for key, value in final_conf.items()]
        logistic_regression_parameters.append(final_conf)
    return logistic_regression_parameters
