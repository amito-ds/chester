from typing import List

import pandas as pd

from mdoel_training.baseline_model import baseline_with_outputs
from mdoel_training.data_preparation import CVData, Parameter
from mdoel_training.models.lgbm_class import lgbm_with_outputs
from mdoel_training.models.logistic_regression import logistic_regression_with_outputs
from mdoel_training.model_input_and_output_classes import ModelResults
from mdoel_training.model_type_detector import ProblemType
from model_compare.compare_messages import compare_models_by_type_and_parameters


class ModelCycle:
    def __init__(self, cv_data: CVData,
                 parameters: List[Parameter] = None,
                 target_col: str = 'target',
                 metric_funcs: List[callable] = None):
        self.cv_data = cv_data
        self.parameters = parameters
        self.target_col = target_col
        self.metric_funcs = metric_funcs
        print("now self.models_results_classification")
        self.models_results_classification = self.running_all_models()

    def get_best_model(self):
        print("getting the best model!")
        models_results_classification = self.models_results_classification
        compare_models_by_type_and_parameters(models_results_classification)  # get init message
        if len(models_results_classification) > 1:
            print("classification results")
            return self.compare_results(models_results_classification)

    def compare_results(self, model_results: list[ModelResults]):
        test_results = \
            [result.results[result.results['type'] == "test"] for result in model_results]
        max_acc = 0
        best_model = None
        for i, result in enumerate(test_results):
            acc = result['accuracy_score'].values[0]
            if acc > max_acc:
                max_acc = acc
                best_model = model_results[i]
        return best_model

    def running_all_models(self) -> list[ModelResults]:
        print("try running running all models")
        print("Considering the inputs, running classification model")
        results1, model1 = baseline_with_outputs(cv_data=self.cv_data, target_col=self.target_col)
        results2, model2, lgbm_parameters = lgbm_with_outputs(
            cv_data=self.cv_data, parameters=self.parameters, target_col=self.target_col)
        results3, model3, logistic_regression_parameters = logistic_regression_with_outputs(
            cv_data=self.cv_data, parameters=self.parameters, target_col=self.target_col)
        model_res1: ModelResults = ModelResults("baseline", model1, pd.DataFrame(results1), [],
                                                predictions=pd.Series())
        model_res2: ModelResults = ModelResults("lgbm", model2, pd.DataFrame(results2), lgbm_parameters,
                                                predictions=pd.Series())
        model_res3: ModelResults = ModelResults("logistic regression", model3, pd.DataFrame(results3),
                                                logistic_regression_parameters, predictions=pd.Series())
        models_results_classification = [model_res1, model_res2, model_res3]


        return models_results_classification


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
