import numpy as np
import pandas as pd
from numpy import float64, int64


class ProblemType:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.is_regression = False
        self.is_classification = False

    def determine_problem_type(self, unique_value_classification_treshold=3):
        # Check the type of target column
        if isinstance(self.y.iloc[0][0], str):
            self.is_classification = True
        elif isinstance(self.y.iloc[0][0], (int64, int, float64, float)):
            # Check the number of unique values in the target column
            unique_values = len(set(np.unique(self.y)))
            if unique_values == 1:
                self.is_regression = False
                self.is_classification = False
            if unique_values > unique_value_classification_treshold:
                self.is_regression = True
            else:
                self.is_regression = True
                self.is_classification = True

    #
    # def check_correlation(self):
    #     # check the correlation between features and target column
    #     corr = np.corrcoef(self.X, self.y)[0, 1]
    #     if abs(corr) > 0.8:
    #         self.is_regression = True
    #     else:
    #         self.is_regression = True
    #         self.is_classification = True
    #
    # def check_multicollinearity(self):
    #     # check for multicollinearity in the data
    #     if np.linalg.cond(self.X) < 1 / np.finfo(float).eps:
    #         self.is_regression = True
    #         self.is_classification = True
    #     else:
    #         self.is_regression = True

    def check_all(self) -> (bool, bool):
        self.determine_problem_type()
        # self.check_correlation()
        # self.check_multicollinearity()
        return self.is_regression, self.is_classification


problem = ProblemType(pd.DataFrame(), pd.DataFrame({'y': ["a", "b", "c"]}))
is_regression, is_classification = problem.check_all()
# print("Is Regression:", is_regression)
# print("Is Classification:", is_classification)


# Example usage
