from chester.model_training.data_preparation import CVData
from chester.model_training.models.chester_models.base_model import BaseModel
from chester.model_training.models.chester_models.catboost.catboost_utils import \
    generate_catboost_configs, catboost_with_outputs, compare_models
from chester.zero_break.problem_specification import DataInfo


class CatboostModel(BaseModel):
    def __init__(self, data_info: DataInfo, cv_data: CVData, num_models_to_compare=10):
        super().__init__(data_info, cv_data, num_models_to_compare)
        self.hp_list = generate_catboost_configs(self.num_models_to_compare)
        print(f"Running {self.num_models_to_compare} catboost models")

    def get_best_model(self):
        models = self.data_info.model_selection_val
        metrics = self.get_metrics_functions()
        if models is None:
            return None
        else:
            models = [model for model in models if "catboost" in model]
            if len(models) == 0:
                return None
            else:
                results = []
                for _ in models:
                    for params in self.hp_list:
                        base_res, model = catboost_with_outputs(
                            cv_data=self.cv_data, target_col=self.cv_data.target_column,
                            parameters=params, metrics=metrics, data_info=self.data_info)
                        results.append((base_res, model))
                best = compare_models(results)
                return best
