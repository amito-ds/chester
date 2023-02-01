import numpy as np
import pandas as pd

from chester.data_loader.webtext_data import load_data_pirates, load_data_king_arthur
from chester.model_training.data_preparation import CVData
from chester.model_training.models.chester_models.base_model import BaseModel
from chester.model_training.models.chester_models.base_model_utils import calculate_metrics_scores
from chester.model_training.models.chester_models.baseline.baseline_utils import train_baseline, predict_baseline, \
    baseline_with_outputs, compare_models
from chester.zero_break.problem_specification import DataInfo


class BaselineModel(BaseModel):
    def __init__(self, data_info: DataInfo, cv_data: CVData, num_models_to_compare=10):
        super().__init__(data_info, cv_data, num_models_to_compare)
        print(f"Running baseline model")

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
                    # results.append(base_res)
                    results.append((base_res, model))
                best = compare_models(results)
                return best

    def calc_model_score(self, model):
        # calculate the score of a given baseline model
        pass

#
# df1 = load_data_pirates().assign(target='chat_logs')
# df2 = load_data_king_arthur().assign(target='pirates')
# df = pd.concat([df1, df2])
# df['target'] = np.random.randint(0, 3, size=len(df))
#
# # Add numerical column
# df["number"] = np.random.uniform(0, 1, df.shape[0])
#
# # Add categorical column
# df["categ"] = 'aaa'
#
# # Add boolean column
# df["booly"] = True
#
# df.drop(columns='text', inplace=True)
# df = df.sample(frac=1).reset_index(drop=True)
#
# # calc data into
# data_info = DataInfo(data=df, target='target')
# data_info.calculate()
# print(data_info)
#
# cv_data = CVData(train_data=df, test_data=None, target_column='target')
# baseline_model = BaselineModel(data_info=data_info, cv_data=cv_data)
# model_results = baseline_model.get_best_model() # returns resultf of the best baseline model
# print(model_results)
