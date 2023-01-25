import random

import numpy as np
from typing import Union

from tcap.model_training.data_preparation import Parameter
from typing import List


class ComplexParameter:
    def __init__(self, name: str, value: Union[tuple, list, int, float]):
        self.name = name
        self.value = value

    def sample(self):
        if type(self.value) == tuple:
            if type(self.value[0]) == int:
                return np.random.randint(self.value[0], self.value[1])
            elif type(self.value[0]) == float:
                return np.random.uniform(self.value[0], self.value[1])
        elif type(self.value) == list:
            return np.random.choice(self.value)


def lgbm_best_practice_hp():
    space = []
    # for continuous hyperparameter
    space.append(ComplexParameter("learning_rate", (0.01, 0.2)))
    space.append(ComplexParameter("n_estimators", (50, 10000)))
    space.append(ComplexParameter("max_depth", (-1, 50)))
    space.append(ComplexParameter("num_leaves", (31, 1024)))
    space.append(ComplexParameter("min_child_samples", (20, 500)))
    space.append(ComplexParameter("subsample", (0.1, 1)))
    space.append(ComplexParameter("colsample_bytree", (0.1, 1)))
    space.append(ComplexParameter("reg_alpha", (0, 10)))
    space.append(ComplexParameter("reg_lambda", (0, 10)))
    # for categorical hyperparameter
    space.append(ComplexParameter("boosting_type", ["gbdt", "dart", "goss"]))
    space.append(ComplexParameter("objective", ["binary", "multiclass", "regression"]))
    space.append(ComplexParameter("metric", ["binary_logloss", "auc", "rmse"]))
    space.append(ComplexParameter("is_unbalance", [True, False]))
    space.append(ComplexParameter("boost_from_average", [True, False]))
    return space


def logistic_regression_best_practice_hp() -> List[ComplexParameter]:
    space = []
    # space.append(ComplexParameter("penalty", ["l1", "l2", "elasticnet"]))
    space.append(ComplexParameter("C", (0.1, 10)))
    space.append(ComplexParameter("max_iter", (100, 1000)))
    return space


from typing import List

from typing import List


def sample_hp_space(hp_space: List[ComplexParameter]) -> List[Parameter]:
    params = []
    penalty = None
    for param in hp_space:
        value = param.sample()
        if param.name == "penalty":
            penalty = value
        params.append(Parameter(param.name, value))

    if penalty == "l1":
        params.append(Parameter("solver", random.choice(["saga"])))
    elif penalty == "l2":
        params.append(Parameter("solver", random.choice(["newton-cg", "lbfgs", "sag"])))
    elif penalty == "elasticnet":
        params.append(Parameter("solver", random.choice(["saga"])))
    elif penalty == "none":
        params.append(Parameter("solver", "newton-cg"))
    return params
