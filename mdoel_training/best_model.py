from typing import List

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from mdoel_training.baseline_model import baseline_with_outputs
from mdoel_training.data_preparation import CVData, Parameter
from mdoel_training.lgbm_class import lgbm_with_outputs
from mdoel_training.logistic_regression import logistic_regression_with_outputs
from mdoel_training.lstm import lstm_with_outputs
from mdoel_training.model_results import ModelResults
from mdoel_training.model_type_detector import ProblemType
from model_compare.compare_messages import compare_models_by_type_and_parameters


class ModelCycle:
    def __init__(self, cv_data: CVData, parameters: List[Parameter] = None, target_col: str = 'target',
                 metric_funcs: List[callable] = None):
        self.cv_data = cv_data
        self.parameters = parameters
        self.target_col = target_col
        self.metric_funcs = metric_funcs
        self.models_results_regression, self.models_results_classification = self.running_all_models()

    def determine_problem_type(self):
        X = self.cv_data.train_data.drop(columns=self.target_col, axis=1)
        y = self.cv_data.train_data[self.target_col]
        return ProblemType(X, y).check_all()

    def compare_model_metrics(self):
        pass

    def get_best_model(self):
        models_results_regression, models_results_classification = self.running_all_models()
        print("models_results_regression", models_results_regression)
        print("models_results_classification", models_results_classification)
        compare_models_by_type_and_parameters(models_results_classification)  # get init message
        # if len(models_results_regression) > 1:
        #     print("regression results")
        #     self.compare_results(models_results_regression)
        # if len(models_results_classification) > 1:
        #     print("classification results")
        #     self.compare_results(models_results_classification)
        pass

    def compare_results(self, model_results: list[ModelResults]):
        test_results = \
            [model_result.aggregate_results()[model_result.aggregate_results()['type'] == "test"] for model_result in
             model_results]
        metric_names = test_results[0].columns
        for metric_name in metric_names:
            is_higher_better = is_metric_higher_better(metric_name)
            if is_higher_better:
                metric_values = [result[metric_name] for result in test_results]
                sorted_models = [model for _, model in sorted(zip(metric_values, model_results), reverse=True)]
            else:
                metric_values = [result[metric_name] for result in test_results]
                sorted_models = [model for _, model in sorted(zip(metric_values, model_results))]
            print(f"The best models for {metric_name} metric are: ")
            for model in sorted_models:
                print(model.model_name)

    def running_all_models(self):
        models_results_classification = None
        models_results_regression = None
        is_regression, is_classification = self.determine_problem_type()
        if is_regression:
            print("Considering the inputs, running regression model")
            pass
        if is_classification:
            label_encoder = LabelEncoder()
            label_encoder = label_encoder.fit(self.cv_data.train_data[self.target_col])
            print("Considering the inputs, running classification model")
            results1, model1 = baseline_with_outputs(
                cv_data=self.cv_data, target_col=self.target_col, metric_funcs=self.metric_funcs)
            results2, model2, lgbm_parameters = lgbm_with_outputs(
                cv_data=self.cv_data, parameters=self.parameters, target_col=self.target_col,
                metric_funcs=self.metric_funcs)
            results3, model3, logistic_regression_parameters = logistic_regression_with_outputs(
                cv_data=self.cv_data, parameters=self.parameters, target_col=self.target_col, metric_funcs=None)
            # results4, model4, parameters, predictions = lstm_with_outputs(
            #     self.cv_data, parameters=self.parameters, target_col=self.target_col,
            #     metric_funcs=self.metric_funcs, label_encoder=label_encoder)
            model_res1: ModelResults = ModelResults("baseline", pd.DataFrame(results1), model1, [],
                                                    predictions=pd.Series())
            model_res2: ModelResults = ModelResults("lgbm", pd.DataFrame(results2), model2, lgbm_parameters,
                                                    predictions=pd.Series())
            model_res3: ModelResults = ModelResults("logistic regression", pd.DataFrame(results3), model3,
                                                    logistic_regression_parameters, predictions=pd.Series())
            models_results_classification = [model_res1, model_res2, model_res3]
        return models_results_regression, models_results_classification


def is_metric_higher_better(metric_name: str) -> bool:
    metric_name = metric_name.lower()
    higher_better_metrics = ["accuracy", "f1", "precision", "recall", "roc", "roc auc", "gini", "r squared", "mape",
                             "mae", "mse"]
    lower_better_metrics = ["rmse", "log loss", "cross entropy", "brier score", "loss"]
    if any(metric in metric_name for metric in higher_better_metrics):
        return True
    elif any(metric in metric_name for metric in lower_better_metrics):
        return False
    else:
        raise ValueError(f"Metric {metric_name} not recognized.")


print(is_metric_higher_better("ROC"))
