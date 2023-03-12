from typing import List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from skopt import gp_minimize

from chester.model_training.data_preparation import CVData, ComplexParameterSet


class HyperparameterOptimization:
    def __init__(self, cv_data: CVData, parameter_set: ComplexParameterSet, metrics: List[Tuple[callable, str]]):
        self.cv_data = cv_data
        self.parameter_set = parameter_set
        self.metrics = metrics

    def optimize(optimization_method='Bayesian', optimization_iterations=20, cv_data=CVData,
                 complex_parameter_set=ComplexParameterSet, target_column='target',
                 metrics_list=[("accuracy", "high")]):
        metric_name, metric_direction = metrics_list[0]
        best_params = None
        best_score = None
        if optimization_method == 'Bayesian':
            # Define the function to optimize
            def logistic_regression_score(params):
                lr = LogisticRegression(C=params[0], penalty=params[1])
                score_list = []
                for train, test in cv_data.splits:
                    X_train, y_train = train_data.iloc[train], train_data[target_column].iloc[train]
                    X_test, y_test = train_data.iloc[test], train_data[target_column].iloc[test]
                    lr.fit(X_train, y_train)
                    y_pred = lr.predict(X_test)
                    score = metric_name(y_test, y_pred)
                    score_list.append(score)
                mean_score = np.mean(score_list)
                return -mean_score if metric_direction == "high" else mean_score

            # Define the parameter space
            param_space = [(0.01, 1), ["l1", "l2"]]

            # Run the optimization
            opt_result = gp_minimize(logistic_regression_score, param_space, n_calls=optimization_iterations)
            best_params = opt_result.x
            best_score = -opt_result.fun if metric_direction == "high" else opt_result.fun
        return best_params, best_score
