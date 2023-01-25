import pandas as pd

from tcap.model_training.data_preparation import CVData
from tcap.model_training.models.scoring import calculate_score_model


class BaselineModel:
    def __init__(self, baseline_value=None, percentile=None):
        self.current_baseline = baseline_value
        self.percentile = percentile

    def fit(self, y):
        if self.current_baseline is None:
            if self.percentile is None:
                self.current_baseline = y.mode()[0]
            else:
                self.current_baseline = y.quantile(self.percentile)
        elif self.percentile is not None:
            self.current_baseline = y.quantile(self.percentile)

    def transform(self, X):
        return pd.Series([self.current_baseline] * len(X))

    def predict(self, X):
        return self.transform(X)

    def fit_transform(self, X, y):
        self.fit(y)
        return self.transform(X)


def train_baseline(X_train, y_train, baseline_value=None, percentile=None):
    """
    Trains a baseline model using the given parameters.
    :param X_train: The training data features (unused in this function)
    :param y_train: The training data labels
    :param baseline_num: A number to use as the baseline (mean, median, mode, or percentile)
    :param percentile: Percentile to use as the baseline if baseline_num is not provided
    :return: A trained baseline model
    """
    model = BaselineModel(baseline_value=baseline_value, percentile=percentile)
    model.fit(y_train)
    # print("baseline model predict", model.transform(X_train))
    return model


def predict_baseline(model, X):
    """
    Makes predictions using a baseline model
    :param model: trained baseline model
    :param X: The data to make predictions on (unused in this function)
    :return: A list of predictions
    """
    return model.transform(X)


def baseline_with_outputs(cv_data: CVData, target_col: str, baseline_num=None, percentile=None):
    results = []
    for i, (train_index, test_index) in enumerate(cv_data.splits):
        X_train, X_test = cv_data.train_data.iloc[train_index], cv_data.train_data.iloc[test_index]
        y_train, y_test = X_train[target_col], X_test[target_col]
        model = train_baseline(X_train, y_train, baseline_num, percentile)
        prediction = predict_baseline(model, X_test)
        prediction_train = predict_baseline(model, X_train)
        scores = calculate_score_model(y_test, prediction)
        results.append({'type': 'test', 'fold': i, **scores})
        scores = calculate_score_model(y_train, prediction_train)
        results.append({'type': 'train', 'fold': i, **scores})
    return results, model
