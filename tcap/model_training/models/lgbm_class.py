# pylint: disable=W0123


def warn(*args, **kwargs):
    pass


import warnings
from collections import defaultdict
import logging

warnings.warn = warn

logging.basicConfig(level=logging.ERROR)

import numpy as np
import pandas as pd
from tcap.model_training.models.model_input_and_output_classes import ModelInput
from tcap.model_training.models.scoring import calculate_score_model
import lightgbm as lgb
from tcap.model_training.data_preparation import CVData, Parameter, ComplexParameter
from sklearn.model_selection import GridSearchCV
from typing import List

default_parameters = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'colsample_bytree': 0.9,
    'subsample': 0.8,
    'subsample_freq': 5,
    'verbose': -1
}

lgbm_class_default_parameters = [
    Parameter('objective', default_parameters['objective']),
    Parameter('boosting_type', default_parameters['boosting_type']),
    Parameter('metric', default_parameters['metric']),
    Parameter('num_leaves', default_parameters['num_leaves']),
    Parameter('learning_rate', default_parameters['learning_rate']),
    Parameter('colsample_bytree', default_parameters['colsample_bytree']),
    Parameter('subsample', default_parameters['subsample']),
    Parameter('subsample_freq', default_parameters['subsample_freq']),
    Parameter('verbose', default_parameters['verbose'])
]


def train_lgbm(X_train, y_train, parameters: list[Parameter]):
    """
    Trains a lightgbm model using the given parameters.
    :param X_train: The training data features
    :param y_train: The training data labels
    :param parameters: A list of dictionaries, each representing a set of hyperparameters
    :return: A trained lightgbm models
    """
    params = {}
    for param in parameters:
        params[param.name] = param.value
    y_train = np.asarray(y_train)

    n_classes = len(np.unique(y_train))
    if n_classes == 2:
        params['objective'] = 'binary'
        params['metric'] = 'binary_logloss'
    else:
        params['objective'] = 'multiclass'
        params['metric'] = 'multi_logloss'
        params['num_class'] = n_classes
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train, verbose=-1)
    return model


def predict_lgbm(model, X):
    """
    Makes predictions using a list of lightgbm models
    :param model: trained lightgbm model
    :param X: The data to make predictions on
    :return: A list of predictions for each model
    """
    return model.predict(X)


def lgbm_with_outputs(cv_data: CVData, parameters: list[Parameter], target_col: str):
    results = []
    if not parameters:
        parameters = lgbm_class_default_parameters
    for i, (train_index, test_index) in enumerate(cv_data.splits):
        X_train, X_test = cv_data.train_data.iloc[train_index], cv_data.train_data.iloc[test_index]
        y_train, y_test = X_train[target_col], X_test[target_col]
        X_train = X_train.drop(columns=[target_col])
        X_test = X_test.drop(columns=[target_col])
        model = train_lgbm(X_train, y_train, parameters)
        train_pred = predict_lgbm(model, X_train)
        test_pred = predict_lgbm(model, X_test)

        train_scores = calculate_score_model(y_train, train_pred)
        test_scores = calculate_score_model(y_test, test_pred)

        results.append(
            {'type': 'train', 'fold': i, **{param.name: param.value for param in parameters}, **train_scores})
        results.append({'type': 'test', 'fold': i, **{param.name: param.value for param in parameters}, **test_scores})

    parameters.append(Parameter("verbose", 1))
    model = train_lgbm(cv_data.train_data.drop(columns=[target_col]), cv_data.train_data[target_col], parameters)
    for i, (train_index, test_index) in enumerate(cv_data.splits):
        X_train, X_test = cv_data.train_data.iloc[train_index], cv_data.train_data.iloc[test_index]
    y_train, y_test = X_train[target_col], X_test[target_col]
    X_train = X_train.drop(columns=[target_col])
    X_test = X_test.drop(columns=[target_col])
    train_pred = predict_lgbm(model, X_train)
    test_pred = predict_lgbm(model, X_test)
    train_scores = calculate_score_model(y_train, train_pred)
    test_scores = calculate_score_model(y_test, test_pred)
    results.append(
        {'type': 'train', 'fold': i + 1, **{param.name: param.value for param in parameters}, **train_scores})
    results.append({'type': 'test', 'fold': i + 1, **{param.name: param.value for param in parameters}, **test_scores})
    return results, model, parameters


def lgbm_grid_search(cv_data: CVData, parameters: List[ComplexParameter], target_col: str,
                     metric_funcs: List[callable] = None):
    if not metric_funcs:
        metric_func = 'accuracy'
    else:
        metric_func = metric_funcs[0]
    if not parameters:
        parameters = lgbm_class_default_parameters
    params = {}
    for param in parameters:
        params[param.name] = param.value
    model = lgb.LGBMClassifier(log_file='lightgbm.log')
    y_train = cv_data.train_data[target_col]
    params['num_class'] = [len(np.unique(y_train))]
    gs = GridSearchCV(model, params, cv=cv_data.splits, scoring=metric_func, return_train_score=True)
    gs.fit(cv_data.train_data.drop(target_col, axis=1), y_train)
    return gs.cv_results_


def lgbm_class_hp(inputs: ModelInput):
    results, _, _ = lgbm_with_outputs(inputs.cv_data, inputs.parameters, inputs.target_col)
    results = pd.DataFrame(results)
    results.drop([p.name for p in inputs.parameters], axis=1, inplace=True)
    results = results.loc[results['type'] == 'test']
    avg_3rd_col = results.iloc[:, 2].mean()
    return avg_3rd_col


def generate_lgbm_configs(k: int) -> List[List[Parameter]]:
    # List of additional configurations to test
    additional_confs = [
        {'num_leaves': 31, 'learning_rate': 0.1},
        {'num_leaves': 31, 'learning_rate': 0.05, 'colsample_bytree': 0.8},
        {'num_leaves': 31, 'learning_rate': 0.01},
        {'num_leaves': 31, 'learning_rate': 0.05, 'colsample_bytree': 0.8, 'subsample': 0.9},
        {'num_leaves': 31, 'learning_rate': 0.05, 'subsample_freq': 2},
        {'num_leaves': 31, 'learning_rate': 0.05, 'subsample': 0.9},
        {'num_leaves': 31, 'learning_rate': 0.05, 'colsample_bytree': 1},
        {'num_leaves': 31, 'learning_rate': 0.05, 'subsample': 0.7},
        {'num_leaves': 31, 'learning_rate': 0.05, 'subsample_freq': 3},
        {'num_leaves': 31, 'learning_rate': 0.05, 'subsample_freq': 1},
    ]
    # List to store the final configurations
    lgbm_class_parameters = []
    for conf in additional_confs[:k]:
        # Create a dictionary to store the final configuration
        final_conf = defaultdict(lambda: None, default_parameters)
        final_conf.update(conf)
        # Convert the dictionary to a list of Parameter objects
        final_conf = [Parameter(key, value) for key, value in final_conf.items()]
        lgbm_class_parameters.append(final_conf)
    return lgbm_class_parameters
