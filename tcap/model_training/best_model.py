from typing import List, Tuple

import pandas as pd

from tcap.model_training.data_preparation import CVData, Parameter
from tcap.model_training.models.baseline_model import baseline_with_outputs
from tcap.model_training.models.lgbm_class import generate_lgbm_configs
from tcap.model_training.models.lgbm_class import lgbm_with_outputs
from tcap.model_training.models.logistic_regression import logistic_regression_with_outputs, \
    generate_logistic_regression_configs
from tcap.model_training.models.model_input_and_output_classes import ModelResults


class CompareModels:
    def __init__(self, models_input: List[Tuple[str, dict]]):
        self.models_input = models_input


class ModelCycle:
    def __init__(self, cv_data: CVData = None,
                 parameters: List[Parameter] = None,
                 target_col: str = 'target',
                 metric_funcs: List[callable] = None,
                 compare_models: CompareModels = None,
                 lgbm_models: int = 10,
                 logistic_regression_models: int = 10):
        self.cv_data = cv_data
        self.parameters = parameters
        self.target_col = target_col
        self.metric_funcs = metric_funcs
        self.compare_models = compare_models
        self.lgbm_models = lgbm_models
        self.logistic_regression_models = logistic_regression_models
        # this one goes last
        self.models_results_classification = self.running_all_models()

    def get_best_model(self):
        print("Choosing a model...\n")
        if self.compare_models is not None:
            models_results_classification = self.run_chosen_models(self.compare_models)
        else:
            models_results_classification = self.models_results_classification
        if len(models_results_classification) > 1:
            return self.compare_results(models_results_classification)
        else:
            return models_results_classification[0]

    def compare_results(self, model_results):
        test_results = \
            [result.results[result.results['type'] == "test"] for result in model_results]
        max_acc = 0
        best_model = None
        for i, result in enumerate(test_results):
            acc = result['accuracy_score'].values[0]
            if acc > max_acc:
                max_acc = acc
                best_model = model_results[i]
        print("Best model: ", best_model.model_name)
        print("With parameters:")
        for param in best_model.parameters:
            print(param.name, "=", param.value)
        return best_model

    def run_chosen_models(self, compare_models: CompareModels):
        models_input = compare_models.models_input
        models_results = []
        for name, value in models_input:
            if name == "lgbm":
                results, model, parameters = lgbm_with_outputs(
                    cv_data=self.cv_data,
                    parameters=convert_param_dict_to_list(value),
                    target_col=self.target_col)
                model_res = ModelResults(name, model, pd.DataFrame(results), parameters, predictions=pd.Series())
                models_results.append(model_res)
            elif name == "logistic regression":
                results, model, parameters = \
                    logistic_regression_with_outputs(self.cv_data,
                                                     target_col=self.target_col,
                                                     parameters=convert_param_dict_to_list(value))
                model_res = ModelResults(name, model, pd.DataFrame(results), parameters, predictions=pd.Series())
                models_results.append(model_res)
            elif name == "baseline":
                results, model = baseline_with_outputs(self.cv_data, self.target_col)
                model_res = ModelResults(name, model, pd.DataFrame(results), [], predictions=pd.Series())
                models_results.append(model_res)
            else:
                print(f"{name} model not recognized")
        return models_results

    def running_all_models(self):
        print("Considering the inputs, running classification model")
        results_baseline, model_baseline = baseline_with_outputs(cv_data=self.cv_data, target_col=self.target_col)
        model_results_baseline: ModelResults = ModelResults("baseline", model_baseline, pd.DataFrame(results_baseline),
                                                            [], predictions=pd.Series())
        lgbm_models_results_organized = []
        if self.lgbm_models > 0:
            lgnm_confs = generate_lgbm_configs(self.lgbm_models)
            print(f"Cpmparing {len(lgnm_confs)} LGBM confs")
            lgbm_models_results = [lgbm_with_outputs(
                cv_data=self.cv_data, parameters=parameters, target_col=self.target_col
            )
                for parameters in lgnm_confs]
            lgbm_models_results_organized = []
            for model_res in lgbm_models_results:
                results, model, lgbm_parameters = model_res
                lgbm_models_results_organized.append(
                    ModelResults("lgbm", model=model, results=pd.DataFrame(results), parameters=lgbm_parameters,
                                 predictions=pd.Series())
                )

        logistic_regression_models_results_organized = []
        if self.logistic_regression_models > 0:
            lr_confs = generate_logistic_regression_configs(self.logistic_regression_models)
            print(f"Cpmparing {len(lr_confs)} Logistic regression confs")
            lr_models_results = [logistic_regression_with_outputs(
                cv_data=self.cv_data, parameters=parameters, target_col=self.target_col
            ) for parameters in lr_confs]

            for model_res in lr_models_results:
                results, model, lr_parameters = model_res
                logistic_regression_models_results_organized.append(
                    ModelResults("logistic regression", model=model, results=pd.DataFrame(results),
                                 parameters=lr_parameters,
                                 predictions=pd.Series())
                )

        return lgbm_models_results_organized + logistic_regression_models_results_organized + [model_results_baseline]


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


def convert_param_dict_to_list(param_dict: dict) -> List[Parameter]:
    param_list = []
    for name, value in param_dict.items():
        param_list.append(Parameter(name, value))
    return param_list
