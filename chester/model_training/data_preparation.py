import os
import random
import sys
from typing import List

path = os.path.abspath("TCAP")
sys.path.append(path)

from sklearn.model_selection import KFold, train_test_split


class CVData:
    def __init__(self, train_data, test_data, target_column, folds=5,
                 split_data=False, split_prop=0.2, split_random_state=42):
        # check if nulls in the target
        null_count = int(train_data[target_column].isnull().sum())
        if null_count > 0:
            null_rows = train_data[train_data[target_column].isnull()].index
            not_null_rows = train_data[~train_data[target_column].isnull()].index
            self.train_data = train_data.loc[not_null_rows]
            self.data_to_predict = train_data.loc[null_rows]
        else:
            self.train_data = train_data
        self.test_data = test_data
        if self.test_data is None:
            if split_data:
                self.train_data, self.test_data = train_test_split(self.train_data, test_size=split_prop,
                                                                   random_state=split_random_state)
        self.target_column = target_column
        self.splits = self.cv_preparation(train_data=self.train_data, test_data=self.test_data, k_fold=folds)

    def format_splits(self):
        formatted_splits = []
        for train_index, test_index in self.splits:
            X_train, y_train = self.train_data.drop(self.target_column, axis=1).iloc[train_index], \
                self.train_data.iloc[train_index][self.target_column]
            X_test, y_test = self.train_data.drop(self.target_column, axis=1).iloc[test_index], \
                self.train_data.iloc[test_index][self.target_column]
            formatted_splits.append((X_train, y_train, X_test, y_test))
        return formatted_splits

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
