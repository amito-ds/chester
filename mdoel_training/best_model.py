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
        # getting results from model training
        models_results_regression, models_results_classification = self.running_all_models()
        compare_models_by_type_and_parameters(models_results_classification)  # get init message
        if len(models_results_classification) > 1:
            print("classification results")
            return self.compare_results(models_results_classification)
        if len(models_results_regression) > 1:
            print("regression results")
            return self.compare_results(models_results_regression)
        pass

    def compare_results(self, model_results: list[ModelResults], decimal_points=3):
        test_results = \
            [model_result.aggregate_results()[model_result.aggregate_results()['type'] == "test"] for model_result in
             model_results]
        metric_names = test_results[0].columns.tolist()
        metric_names.pop(0)
        metric_names.pop(0)
        print("metric_names", metric_names)
        for i, metric_name in enumerate(metric_names):
            is_higher_better = is_metric_higher_better(metric_name)
            if is_higher_better:
                metric_values = [round(result[metric_name].values[0], decimal_points) for result in test_results]
                sorted_models = [model for _, model in sorted(zip(metric_values, model_results), reverse=True)]
                if i == 0:
                    best_model = sorted_models[0]
            else:
                metric_values = [round(result[metric_name].values[0], decimal_points) for result in test_results]
                sorted_models = [model for _, model in sorted(zip(metric_values, model_results))]
                if i == 0:
                    best_model = sorted_models[0]
            print(f"The best models for {metric_name} metric are: ")
            for i, model in enumerate(sorted_models):
                print(f"{i + 1}. {model.model_name} with value: {metric_values[i]}")
        return best_model

    def running_all_models(self) -> (list[ModelResults], list[ModelResults]):
        models_results_classification: list[ModelResults] = []
        models_results_regression: list[ModelResults] = []
        is_regression, is_classification = self.determine_problem_type()
        if is_regression:
            print("Considering the inputs, running regression model")
            pass
        if is_classification:
            # label_encoder = LabelEncoder()
            # label_encoder = label_encoder.fit(self.cv_data.train_data[self.target_col])
            print("Considering the inputs, running classification model")
            results1, model1 = baseline_with_outputs(
                cv_data=self.cv_data, target_col=self.target_col, metric_funcs=self.metric_funcs)
            results2, model2, lgbm_parameters = lgbm_with_outputs(
                cv_data=self.cv_data, parameters=self.parameters, target_col=self.target_col,
                metric_funcs=self.metric_funcs)
            results3, model3, logistic_regression_parameters = logistic_regression_with_outputs(
                cv_data=self.cv_data, parameters=self.parameters, target_col=self.target_col, metric_funcs=None)
            model_res1: ModelResults = ModelResults("baseline", model1, pd.DataFrame(results1), [],
                                                    predictions=pd.Series())
            model_res2: ModelResults = ModelResults("lgbm", model2, pd.DataFrame(results2), lgbm_parameters,
                                                    predictions=pd.Series())
            model_res3: ModelResults = ModelResults("logistic regression", model3, pd.DataFrame(results3),
                                                    logistic_regression_parameters, predictions=pd.Series())
            models_results_classification = [model_res1, model_res2, model_res3]
            # models_results_classification = [model_res2]
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

# print(is_metric_higher_better("ROC"))
