import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support

from mdoel_training.data_preparation import CVData, Parameter, ComplexParameter
from typing import List
from sklearn.preprocessing import LabelBinarizer

import pandas as pd

from mdoel_training.model_input_and_output_classes import ModelInput
from mdoel_training.models.scoring import calculate_score_model

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


def logistic_regression_with_outputs(cv_data: CVData, target_col: str, parameters: list[Parameter] = None,
                                     metric_funcs: List[callable] = None):
    results = []
    if not parameters:
        parameters = logistic_regression_default_parameters

    if not metric_funcs:
        metric_funcs = [accuracy_score, precision_recall_fscore_support, recall_score, f1_score]
    for i, (train_index, test_index) in enumerate(cv_data.splits):
        model = LogisticRegression()
        X_train, X_test = cv_data.train_data.iloc[train_index], cv_data.train_data.iloc[test_index]
        y_train, y_test = X_train[target_col], X_test[target_col]
        X_train = X_train.drop(columns=[target_col])
        X_test = X_test.drop(columns=[target_col])
        for param in parameters:
            setattr(model, param.name, param.value)
        print([(p.name, p.value) for p in parameters])
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        # label handeling
        # label_binarizer = LabelBinarizer()
        # y_train = label_binarizer.fit_transform(y_train)
        # y_test = label_binarizer.transform(y_test)
        # train_pred = label_binarizer.transform(train_pred)
        # test_pred = label_binarizer.transform(test_pred)

        # print("y_train", y_train)
        # print("train_pred", train_pred)

        train_scores = calculate_score_model(y_train, train_pred)
        test_scores = calculate_score_model(y_test, test_pred)

        results.append(
            {'type': 'train', 'fold': i, **{param.name: param.value for param in parameters}, **train_scores})
        results.append({'type': 'test', 'fold': i, **{param.name: param.value for param in parameters}, **test_scores})
    model.fit(cv_data.train_data.drop(columns=[target_col]), cv_data.train_data[target_col])
    return results, model, parameters


def logistic_regression_hp(inputs: ModelInput):
    results, _, _ = logistic_regression_with_outputs(
        inputs.cv_data, target_col=inputs.target_col, parameters=inputs.parameters)
    results = pd.DataFrame(results)
    results.drop([p.name for p in inputs.parameters], axis=1, inplace=True)
    results = results.loc[results['type'] == 'test']
    print(inputs.parameters)
    avg_3rd_col = results.iloc[:, 2].mean()
    return avg_3rd_col
