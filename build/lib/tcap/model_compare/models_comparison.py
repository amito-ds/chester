from typing import List

import pandas as pd

from tcap.model_training.data_preparation import Parameter
from tcap.model_training.models.model_input_and_output_classes import ModelResults
from model_compare.compare_messages import compare_models_by_type_and_parameters
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class ModelComparison:
    def __init__(self, model_results: list[ModelResults]):
        self.model_results = model_results

    def convert_to_model_tuples(self) -> List[tuple]:
        model_tuples = []
        for model_result in self.model_results:
            param_dict = {}
            for param in model_result.parameters:
                param_dict[param.name] = param.value
            model_tuples.append((model_result.model_name, None, model_result.model, param_dict))
        return model_tuples

    def print_top_message(self):
        compare_models_by_type_and_parameters(self.convert_to_model_tuples())


# Create some sample model results
param1 = Parameter("penalty", "l1")
param2 = Parameter("C", 0.1)
model_results = [ModelResults("Logistic Regression", LogisticRegression(), pd.DataFrame(), [param1, param2], pd.Series()),
                 ModelResults("Random Forest", RandomForestClassifier(), pd.DataFrame(), [param1, param2], pd.Series())]

# Create an instance of the ModelComparison class and call the print_top_message function
# comparison = ModelComparison(model_results)
# comparison.print_top_message()
