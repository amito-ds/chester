from collections import defaultdict
from typing import List

import pandas as pd

from chester.model_training.data_preparation import CVData
from chester.model_training.data_preparation import Parameter
from chester.model_training.models.chester_models.base_model_utils import calculate_metrics_scores
from chester.model_training.models.chester_models.base_model_utils import is_metric_higher_is_better
from chester.model_training.models.chester_models.catboost.catboost_model import CatboostModel
from chester.model_training.models.chester_models.logistic_regression.logistic_regression_model import \
    LogisticRegressionModel


def train_catboost(X_train, y_train, parameters: list, problem_type: str):
    """
    Trains a baseline model using the given parameters.
    :param X_train: The training data features (unused in this function)
    :param y_train: The training data labels
    :param parameters: parameters list
    :return: A trained baseline model
    """
    model = CatboostModel(parameters, problem_type)
    model.fit(X_train, y_train)
    return model


def predict_catboost(model, X):
    """
    Makes predictions using a baseline model
    :param model: trained baseline model
    :param X: The data to make predictions on (unused in this function)
    :return: A list of predictions
    """
    return model.transform(X)


def catboost_with_outputs(cv_data: CVData,
                          metrics: list,
                          target_col: str,
                          parameters: list,
                          problem_type=None,
                          ):
    results = []
    for i, (X_train, y_train, X_test, y_test) in enumerate(cv_data.format_splits()):
        model = train_catboost(X_train, y_train, parameters=parameters, problem_type=problem_type)
        prediction = predict_catboost(model, X_test)
        prediction_train = predict_catboost(model, X_train)
        # print("prediction_train", prediction_train)
        # print(metrics)
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
    'iterations': 1000,
    'learning_rate': 0.03,
    'depth': 6,
    'random_seed': 42,
    'verbose': False
}

# For regression
regression_parameters = default_parameters.copy()
regression_parameters.update({'loss_function': 'RMSE', 'evaluation_metric': 'RMSE'})

# For binary classification
binary_classification_parameters = default_parameters.copy()
binary_classification_parameters.update({'loss_function': 'Logloss', 'evaluation_metric': 'AUC'})

# For multiclass classification
multiclass_classification_parameters = default_parameters.copy()
multiclass_classification_parameters.update({'loss_function': 'MultiClass', 'evaluation_metric': 'F1'})


def generate_catboost_configs(k: int, problem_type: str) -> List[List[Parameter]]:
    # default_parameters should be by problem type
    catboost_default_parameters = default_parameters
    if problem_type == "Binary regression" or problem_type == "Regression":
        catboost_default_parameters = regression_parameters
    elif problem_type == "Binary classification":
        catboost_default_parameters = binary_classification_parameters
    elif problem_type == "Multiclass classification":
        catboost_default_parameters = multiclass_classification_parameters

    # List of additional configurations to test
    additional_confs = [
        {**catboost_default_parameters, 'iterations': 500, 'learning_rate': 0.01, 'depth': 4},
        {**catboost_default_parameters, 'iterations': 2000, 'learning_rate': 0.05, 'depth': 8},
        {**catboost_default_parameters, 'iterations': 1000, 'learning_rate': 0.03, 'depth': 6},
        {**catboost_default_parameters, 'iterations': 1500, 'learning_rate': 0.02, 'depth': 5},
        {**catboost_default_parameters, 'iterations': 750, 'learning_rate': 0.04, 'depth': 7},
        {**catboost_default_parameters, 'iterations': 2500, 'learning_rate': 0.01, 'depth': 4},
        {**catboost_default_parameters, 'iterations': 500, 'learning_rate': 0.06, 'depth': 8},
        {**catboost_default_parameters, 'iterations': 1500, 'learning_rate': 0.05, 'depth': 5},
        {**catboost_default_parameters, 'iterations': 1000, 'learning_rate': 0.04, 'depth': 7},
        {**catboost_default_parameters, 'iterations': 2000, 'learning_rate': 0.02, 'depth': 6},
        {**catboost_default_parameters, 'iterations': 750, 'learning_rate': 0.03, 'depth': 5},
        {**catboost_default_parameters, 'iterations': 2500, 'learning_rate': 0.04, 'depth': 7},
        {**catboost_default_parameters, 'iterations': 1500, 'learning_rate': 0.06, 'depth': 6},
        {**catboost_default_parameters, 'iterations': 1000, 'learning_rate': 0.05, 'depth': 8},
        {**catboost_default_parameters, 'iterations': 500, 'learning_rate': 0.03, 'depth': 5},
        {**catboost_default_parameters, 'iterations': 2000, 'learning_rate': 0.01, 'depth': 7},
        {**catboost_default_parameters, 'iterations': 750, 'learning_rate': 0.06, 'depth': 6},
        {**catboost_default_parameters, 'iterations': 2500, 'learning_rate': 0.02, 'depth': 8},
        {**catboost_default_parameters, 'iterations': 1500, 'learning_rate': 0.04, 'depth': 5}
    ]

    # List to store the final configurations
    catboost_parameters = []
    for conf in additional_confs[:k]:
        # Create a dictionary to store the final configuration
        final_conf = defaultdict()
        final_conf.update(conf)
        # Convert the dictionary to a list of Parameter objects
        final_conf = [Parameter(key, value) for key, value in final_conf.items()]
        catboost_parameters.append(final_conf)
    return catboost_parameters
