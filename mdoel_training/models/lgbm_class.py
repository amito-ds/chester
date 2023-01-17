import logging

import numpy as np

from mdoel_training.model_input_and_output_classes import ModelInput
from mdoel_training.model_utils import organize_results

logging.getLogger("lightgbm").setLevel(logging.ERROR)
import lightgbm as lgb

from typing import List

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, recall_score, f1_score
from sklearn.preprocessing import LabelBinarizer

from mdoel_training.data_preparation import CVData, Parameter, ComplexParameter
from typing import List, Dict
from itertools import product
from sklearn.model_selection import GridSearchCV

default_parameters = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

lgbm_class_default_parameters = [
    Parameter('objective', default_parameters['objective']),
    Parameter('boosting_type', default_parameters['boosting_type']),
    Parameter('metric', default_parameters['metric']),
    Parameter('num_leaves', default_parameters['num_leaves']),
    Parameter('learning_rate', default_parameters['learning_rate']),
    Parameter('feature_fraction', default_parameters['feature_fraction']),
    Parameter('bagging_fraction', default_parameters['bagging_fraction']),
    Parameter('bagging_freq', default_parameters['bagging_freq']),
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
        params['num_class'] = n_classes
    print(params)
    print(len(np.unique(y_train)))
    print(type(y_train))
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


def score_lgbm(y, prediction, metric_funcs: List[callable]):
    """
    Calculates evaluation metrics for the predictions
    :param y: The true labels
    :param prediction: predictions
    :param metric_funcs: A list of evaluation metric functions
    :return: A dictionary of metric scores for each model
    """
    label_binarizer = LabelBinarizer()
    y = label_binarizer.fit_transform(y)
    pred = label_binarizer.transform(prediction)
    scores = {metric.__name__: metric(y, pred) for metric in metric_funcs}
    return scores


def lgbm_with_outputs(cv_data: CVData, parameters: list[Parameter], target_col: str,
                      metric_funcs: List[callable] = None):
    results = []
    if not metric_funcs:
        metric_funcs = [accuracy_score, precision_recall_fscore_support, recall_score, f1_score]
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

        train_scores = score_lgbm(y_train, train_pred, metric_funcs)
        test_scores = score_lgbm(y_test, test_pred, metric_funcs)

        results.append(
            {'type': 'train', 'fold': i, **{param.name: param.value for param in parameters}, **train_scores})
        results.append({'type': 'test', 'fold': i, **{param.name: param.value for param in parameters}, **test_scores})

    model = train_lgbm(cv_data.train_data.drop(columns=[target_col]), cv_data.train_data[target_col], parameters)
    for i, (train_index, test_index) in enumerate(cv_data.splits):
        X_train, X_test = cv_data.train_data.iloc[train_index], cv_data.train_data.iloc[test_index]
    y_train, y_test = X_train[target_col], X_test[target_col]
    X_train = X_train.drop(columns=[target_col])
    X_test = X_test.drop(columns=[target_col])
    train_pred = predict_lgbm(model, X_train)
    test_pred = predict_lgbm(model, X_test)
    train_scores = score_lgbm(y_train, train_pred, metric_funcs)
    test_scores = score_lgbm(y_test, test_pred, metric_funcs)
    results.append({'type': 'train', 'fold': i, **{param.name: param.value for param in parameters}, **train_scores})
    results.append({'type': 'test', 'fold': i, **{param.name: param.value for param in parameters}, **test_scores})
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
    model = lgb.LGBMClassifier()
    y_train = cv_data.train_data[target_col]
    params['num_class'] = [len(np.unique(y_train))]
    gs = GridSearchCV(model, params, cv=cv_data.splits, scoring=metric_func, return_train_score=True)
    print(np.unique(cv_data.train_data[target_col]))
    gs.fit(cv_data.train_data.drop(target_col, axis=1), y_train)
    return gs.cv_results_


def lgbm_class_hp(inputs: ModelInput):
    results, _, _ = lgbm_with_outputs(inputs.cv_data, inputs.parameters, inputs.target_col)
    results = organize_results(results)
    results.drop([p.name for p in inputs.parameters], axis=1, inplace=True)
    results = results.loc[results['type'] == 'test']
    print(results)
    avg_3rd_col = results.iloc[:, 2].mean()
    return avg_3rd_col
