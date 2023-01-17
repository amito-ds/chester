import pandas as pd
from typing import List

from mdoel_training.data_preparation import Parameter, CVData
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression

from mdoel_training.model_utils import organize_results


class ModelResults:
    def __init__(self, model_name: str, model, results: pd.DataFrame, parameters: List, predictions: pd.Series):
        self.model_name = model_name
        self.model = model
        self.results = results
        self.parameters = parameters
        self.predictions = predictions

    def aggregate_results(self):
        # Remove columns that appear in parameters
        # param_cols = [param.name for param in self.parameters]
        # self.results.drop(columns=param_cols, inplace=True)

        # Aggregate over folds
        aggregate_df = self.results.groupby(['type']).mean()
        aggregate_df.reset_index(inplace=True)

        return aggregate_df


class ModelInput:
    def __init__(self, cv_data: CVData, parameters: List[Parameter], target_col: str):
        self.cv_data = cv_data
        self.parameters = parameters
        self.target_col = target_col
