from typing import List
import pandas as pd
from tcap.model_training.data_preparation import Parameter, CVData


class ModelResults:
    def __init__(self, model_name: str, model, results: pd.DataFrame, parameters: List, predictions: pd.Series):
        self.model_name = model_name
        self.model = model
        self.results = results
        self.parameters = parameters
        self.predictions = predictions

    def aggregate_results(self):
        # Aggregate over folds
        aggregate_df = self.results.groupby(['type']).mean()
        aggregate_df.reset_index(inplace=True)

        return aggregate_df


class ModelInput:
    def __init__(self, cv_data: CVData = None, parameters: List[Parameter] = None, target_col: str = 'target',
                 name: str = None):
        self.cv_data = cv_data
        self.parameters = parameters
        self.target_col = target_col
        self.name = name
