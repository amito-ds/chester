from typing import List, Tuple, Union

from mdoel_training.data_preparation import CVData, Parameter
from mdoel_training.model_type_detector import ProblemType


class ModelCycle:
    def __init__(self, cv_data: CVData, parameters: List[Parameter], target_col: str = 'target',
                 metric_funcs: List[callable] = None):
        self.cv_data = cv_data
        self.parameters = parameters
        self.target_col = target_col
        self.metric_funcs = metric_funcs

    def determine_problem_type(self):
        X = self.cv_data.train_data.drop(self.target_col, axis=1)
        y = self.cv_data.train_data[self.target_col]
        return ProblemType(X, y).check_all()

    def get_best_model(self):
        is_regression, is_classification = self.determine_problem_type()
        if (is_regression):
            print("Considering the inputs, running regression model")
            pass
        if (is_classification):
            print("Considering the inputs, running classification model")
            pass

    def train_and_evaluate(self):
        self.determine_problem_type()
        best_model = self.get_best_model()
        # train the best model with the entire train dataset
        # evaluate the model using the test dataset
        pass
