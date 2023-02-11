from chester.model_training.data_preparation import CVData
from chester.model_training.models.chester_models.base_model import BaseModel
from chester.model_training.models.chester_models.linear_regression.linear_regression_utils import \
    generate_linear_regression_configs, linear_regression_with_outputs, compare_models
from chester.zero_break.problem_specification import DataInfo


class LinearRegressionModel(BaseModel):
    def __init__(self, data_info: DataInfo, cv_data: CVData, num_models_to_compare=15, best_practice_prob=0.33):
        super().__init__(data_info, cv_data, num_models_to_compare, best_practice_prob)
        self.hp_list = generate_linear_regression_configs(k=self.num_models_to_compare,
                                                          best_practice_prob=self.best_practice_prop)
        print(f"Running {self.num_models_to_compare} linear models")

    def get_best_model(self):
        models = self.data_info.model_selection_val
        metrics = self.get_metrics_functions()
        if models is None:
            return None
        else:
            models = [model for model in models if "catboost" in model]
            # print("models", models)
            if len(models) == 0:
                return None
            else:
                results = []
                for _ in models:
                    for params in self.hp_list:
                        base_res, model = linear_regression_with_outputs(
                            cv_data=self.cv_data, target_col=self.cv_data.target_column,
                            parameters=params, metrics=metrics, data_info=self.data_info)
                        results.append((base_res, model))
                best = compare_models(results)
                return best
