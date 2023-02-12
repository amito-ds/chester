from chester.model_training.data_preparation import CVData
from chester.model_training.models.chester_models.base_model import BaseModel
from chester.model_training.models.chester_models.baseline.baseline_utils import baseline_with_outputs, compare_models
from chester.zero_break.problem_specification import DataInfo


class BaselineModel(BaseModel):
    def __init__(self, data_info: DataInfo, cv_data: CVData, num_models_to_compare=10):
        super().__init__(data_info, cv_data, num_models_to_compare)

    def get_best_model(self):
        models = self.data_info.model_selection_val
        metrics = self.get_metrics_functions()
        if models is None:
            return None
        else:
            baseline_models = [model.split("-")[1] for model in models if "baseline" in model]
            if len(baseline_models) == 0:
                return None
            else:
                results = []
                for baseline_model in baseline_models:
                    median_baseline = None
                    if baseline_model == 'median':
                        median_baseline = 0.5

                    average_baseline = False
                    if baseline_model == 'average':
                        average_baseline = True

                    baseline_value = None
                    if 'value' in baseline_model:
                        baseline_value = baseline_model.split(" ")[1]
                    base_res, model = baseline_with_outputs(
                        cv_data=self.cv_data, target_col=self.cv_data.target_column,
                        baseline_value=baseline_value, avg_baseline=average_baseline, median_baseline=median_baseline,
                        metrics=metrics)
                    results.append((base_res, model))
                best = compare_models(results)
                return best

    def calc_model_score(self, model):
        # calculate the score of a given baseline model
        pass


