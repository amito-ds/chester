from chester.model_training.data_preparation import CVData
from chester.model_training.models.chester_models.base_model import BaseModel
from chester.model_training.models.chester_models.logistic_regression.logistic_regression_utils import \
    generate_logistic_regression_configs, logistic_regression_with_outputs, compare_models
from chester.zero_break.problem_specification import DataInfo


class LogisticRegressionModel(BaseModel):
    def __init__(self, data_info: DataInfo, cv_data: CVData, num_models_to_compare=9, best_practice_prob=0.33):
        super().__init__(data_info, cv_data, num_models_to_compare)
        self.hp_list = generate_logistic_regression_configs(k=self.num_models_to_compare,
                                                            best_practice_prob=best_practice_prob)
        print(f"Running {self.num_models_to_compare} Logistic Regression models")

    def get_best_model(self):
        models = self.data_info.model_selection_val
        metrics = self.get_metrics_functions()
        if models is None:
            return None
        else:
            models = [model for model in models if "logistic" in model]
            if len(models) == 0:
                return None
            else:
                results = []
                for _ in models:
                    for params in self.hp_list:
                        base_res, model = logistic_regression_with_outputs(
                            cv_data=self.cv_data, target_col=self.cv_data.target_column,
                            parameters=params, metrics=metrics, data_info=self.data_info,
                            problem_type=self.data_info.problem_type_val)
                        results.append((base_res, model))
                best = compare_models(results)
                return best

    def calc_model_score(self, model):
        # calculate the score of a given baseline model
        pass
