from collections import defaultdict
from typing import List
import pandas as pd
from tcap.model_training.data_preparation import CVData, Parameter
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_recall_fscore_support
from tcap.model_training.models.model_input_and_output_classes import ModelInput
from tcap.model_training.models.scoring import calculate_score_model

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


def logistic_regression_with_outputs(cv_data: CVData, target_col: str, parameters=None,
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


from typing import List
from sklearn.linear_model import LogisticRegression


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
