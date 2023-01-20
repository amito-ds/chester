import os
import random
import sys
from typing import List


path = os.path.abspath("TCAP")
sys.path.append(path)

from sklearn.model_selection import KFold


class CVData:
    def __init__(self, train_data, test_data, folds=5):
        self.train_data = train_data
        self.test_data = test_data
        self.splits = self.cv_preparation(train_data=train_data, test_data=test_data, k_fold=folds)

    @staticmethod
    def cv_preparation(train_data, test_data=None, k_fold=0):
        if k_fold == 0:
            return None
        elif k_fold > 0:
            kf = KFold(n_splits=k_fold)
            splits = [(train, test) for train, test in kf.split(train_data)]
            return splits
        else:
            raise ValueError("k_fold must be a positive integer")


class Parameter:
    def __init__(self, name, value):
        self.name = name
        self.value = value


class ComplexParameter:

    def __init__(self, name: str, options: List):
        self.name = name
        self.options = options
        self.validate_options()

    def validate_options(self):
        # check if options is a valid type (int, float, tuple, list)
        pass

    def sample(self):
        if isinstance(self.options, int):
            return Parameter(self.name, random.randint(0, self.options))
        elif isinstance(self.options, float):
            return Parameter(self.name, random.uniform(0, self.options))
        elif isinstance(self.options, tuple):
            return Parameter(self.name, random.uniform(self.options[0], self.options[1]))
        elif isinstance(self.options, list):
            return Parameter(self.name, random.choice(self.options))


class ComplexParameterSet:
    def __init__(self, parameters: List[ComplexParameter]):
        self.parameters = parameters

    def sample(self):
        return [parameter.sample() for parameter in self.parameters]


def print_report(parameters: List[Parameter]):
    for param in parameters:
        print(f"{param.name}: {param.value}")