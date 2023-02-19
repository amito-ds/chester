from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, recall_score, f1_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder

from chester.model_training.data_preparation import CVData, Parameter


# from tensorflow.keras.layers import LSTM, Dense


def get_default_parameters(X_train: pd.DataFrame):
    default_parameters = {
        'input_shape': (X_train.shape[1], 1),
        'output_dim': 10,
        'recurrent_dropout': 0.1,
        'dropout': 0.1,
        'optimizer': 'adam',
        'loss': 'binary_crossentropy',
        'metrics': ['accuracy']
    }
    lstm_default_parameters = [
        Parameter(name, default_parameters[name]) for name in default_parameters
    ]

    return lstm_default_parameters


def train_lstm(X_train, y_train, parameters):
    """
    Trains a LSTM model using the given parameters.
    :param X_train: The training data features
    :param y_train: The training data labels
    :param parameters: A list of dictionaries, each representing a set of hyperparameters
    :return: A list of trained LSTM models
    """
    if not parameters:
        parameters = get_default_parameters(X_train)
    params = {}
    for param in parameters:
        params[param.name] = param.value
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.LSTM(params['output_dim'], input_shape=params['input_shape'],
                             recurrent_dropout=params['recurrent_dropout'],
                             dropout=params['dropout']))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=params['optimizer'], metrics=['accuracy'])

    # convert y_train to numerical values
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    model.fit(X_train, y_train)
    return model, label_encoder


def predict_lstm(model, X):
    """
    Makes predictions using a list of LSTM models
    :param model: trained LSTM model
    :param X: The data to make predictions on
    :return: A list of predictions for each model
    """
    return model.predict(X)


def score_lstm(y, prediction, metric_funcs: List[callable]):
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


def lstm_with_outputs(cv_data: CVData, parameters, target_col: str,
                      metric_funcs: List[callable] = None, label_encoder=None):
    results = []
    if not metric_funcs:
        metric_funcs = [accuracy_score, precision_recall_fscore_support, recall_score, f1_score]
    if not parameters:
        parameters = get_default_parameters(cv_data.train_data.drop(columns=[target_col]))
    for p in parameters:
        print(p.name, p.value)
    pass
    for i, (train_index, test_index) in enumerate(cv_data.splits):
        X_train, X_test = cv_data.train_data.iloc[train_index], cv_data.train_data.iloc[test_index]
        y_train, y_test = X_train[target_col], X_test[target_col]
        X_train = X_train.drop(columns=[target_col])
        X_test = X_test.drop(columns=[target_col])
        # reshape input data to 3D
        X_train = X_train.to_numpy()
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.to_numpy()
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        model, label_encoder = train_lstm(X_train, y_train, parameters)
        train_pred = predict_lstm(model, X_train)
        test_pred = predict_lstm(model, X_test)
        train_pred = np.round(train_pred)
        test_pred = np.round(test_pred)
        train_scores = score_lstm(y_train, train_pred, metric_funcs)
        test_scores = score_lstm(y_test, test_pred, metric_funcs)
        results.append(
            {'type': 'train', 'fold': i, **{param.name: param.value for param in parameters}, **train_scores})
        results.append({'type': 'test', 'fold': i, **{param.name: param.value for param in parameters}, **test_scores})
    model, label_encoder = train_lstm(cv_data.train_data.drop(columns=[target_col]), cv_data.train_data[target_col],
                                      parameters)
    return results, model, parameters, model.predict(cv_data.train_data.drop(columns=[target_col]))
