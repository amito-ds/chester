from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

# from data_loader.webtext_data import load_data_chat_logs
# from keras.models import Sequential
# from keras.layers import LSTM, Dense
# from sklearn.preprocessing import MinMaxScaler
import tensorflow
def create_logistic_regression(penalty='l2', C=1.0, class_weight=None):
    logreg = LogisticRegression(penalty=penalty, C=C, class_weight=class_weight)
    return logreg


from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures


def create_regression_model(train_data, test_data=None, target_col='target', ridge=False, lasso=False,
                            elastic_net=False, alpha=5, l1_ratio=0.5, interaction_power=0):
    """
    Create linear regression model based on the provided flags and parameters.
    It will return the unfitted regression model and the fitted poly instance, in case interaction power > 0.

    Parameters
    ----------
    train_data : pandas DataFrame
        The dataframe containing the training data.
    test_data : pandas DataFrame, optional
        The dataframe containing the test data. The default is None.
    target_col : str, optional
        The name of the target column in the dataframe. The default is 'target'.
    ridge : bool, optional
        Flag to create a Ridge regression model. The default is False.
    lasso : bool, optional
        Flag to create a Lasso regression model. The default is False.
    elastic_net : bool, optional
        Flag to create a ElasticNet regression model. The default is False.
    alpha : float, optional
        Regularization strength; higher values specify stronger regularization. The default is 5.
    l1_ratio : float, optional
        The mixing parameter that controls the balance between L1 and L2 penalties. The default is 0.5.
    interaction_power : int, optional
        The maximum degree of the polynomial features. The default is 0, meaning no interaction.

    Returns
    -------
    tuple of reg and poly
        tuple containing:
            reg : object
                The unfitted linear regression model
            poly : object
                The fitted PolynomialFeatures instance, in case interaction power > 0 else None
    """

    X_train = train_data.drop(columns=target_col)
    y_train = train_data[target_col]

    poly = None
    if interaction_power > 0:
        poly = PolynomialFeatures(degree=interaction_power, include_bias=False)
        X_train = poly.fit_transform(X_train)
    if elastic_net:
        reg = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    elif ridge:
        reg = Ridge(alpha=alpha)
    elif lasso:
        reg = Lasso(alpha=alpha)
    else:
        reg = LinearRegression()
    reg.fit(X_train, y_train)
    return reg, poly if interaction_power > 0 else None


import pandas as pd


class BaselineModel:
    def __init__(self, baseline_num=None, percentile=None):
        self.current_baseline = baseline_num
        self.percentile = percentile

    def fit(self, y):
        if self.current_baseline is None:
            self.current_baseline = y.mean()
        elif self.current_baseline == 'median':
            self.current_baseline = y.median()
        elif self.current_baseline == 'mode':
            self.current_baseline = y.mode()[0]
        elif self.percentile is not None:
            self.current_baseline = y.quantile(self.percentile)

    def transform(self, X):
        return pd.Series([self.current_baseline] * len(X))

    def fit_transform(self, X, y):
        self.fit(y)
        return self.transform(X)



import numpy as np


def create_LSTM_model(train_data, test_data=None, target_col='target' , net_params=None):
    """
    Creates a LSTM model.
    :param X_train: The feature matrix of the training data.
    :param y_train: The target vector of the training data.
    :param net_params: A dictionary of network parameters. (default=None)
    :return: A trained LSTM model, and the trained scaler for the features
    """
    scaler = MinMaxScaler()

    lstm = Sequential()
    lstm.add(LSTM(**net_params))
    lstm.add(Dense(1))
    lstm.compile(loss='mean_squared_error', optimizer='adam')
    return lstm, scaler


if __name__ == '__main__':
    print(1)
    # df = load_data_chat_logs()
    # print(df.columns)
    # logreg = create_regression_model(train_data=df, target_col='text')
    # logreg = create_LSTM_model(train_data=df, target_col='text')
