import pandas as pd
from typing import List

from mdoel_training.data_preparation import Parameter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression


class ModelResults:
    def __init__(self, model_name: str, model, results: pd.DataFrame, parameters: List, predictions: pd.Series):
        self.model_name = model_name
        self.model = model
        self.results = results
        self.parameters = parameters
        self.predictions = predictions
