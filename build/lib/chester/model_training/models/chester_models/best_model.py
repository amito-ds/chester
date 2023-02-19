from chester.model_training.data_preparation import CVData
from chester.model_training.models.chester_models import best_catboost
from chester.model_training.models.chester_models import best_logistic_regression
from chester.model_training.models.chester_models.base_model import BaseModel
from chester.model_training.models.chester_models.best_baseline_model import BaselineModel
from chester.model_training.models.chester_models.best_linear_regression import LinearRegressionModel
from chester.model_training.models.chester_models.catboost.catboost_utils import compare_best_models
from chester.zero_break.problem_specification import DataInfo


class BestModel(BaseModel):
    def __init__(self, data_info: DataInfo, cv_data: CVData, num_models_to_compare=15, best_practice_prob=0.33):
        super().__init__(data_info, cv_data, num_models_to_compare, best_practice_prob)

    def get_best_model(self):
        models_type = self.data_info.model_selection_val
        models = []
        for model_type in models_type:
            if 'baseline' in model_type:
                base_res, model = BaselineModel(data_info=self.data_info,
                                                cv_data=self.cv_data,
                                                num_models_to_compare=3
                                                ).get_best_model()
                models.append((base_res, model))
            elif 'logistic' in model_type:
                base_res, model = best_logistic_regression.LogisticRegressionModel(
                    data_info=self.data_info,
                    cv_data=self.cv_data,
                    num_models_to_compare=self.num_models_to_compare,
                    best_practice_prob=self.best_practice_prop
                ).get_best_model()
                models.append((base_res, model))
            elif 'catboost' in model_type:
                base_res, model = best_catboost.CatboostModel(
                    data_info=self.data_info,
                    cv_data=self.cv_data,
                    num_models_to_compare=self.num_models_to_compare
                ).get_best_model()
                models.append((base_res, model))
            elif 'linear' in model_type:
                base_res, model = LinearRegressionModel(
                    data_info=self.data_info,
                    cv_data=self.cv_data,
                    num_models_to_compare=self.num_models_to_compare,
                    best_practice_prob=self.best_practice_prop
                ).get_best_model()
                models.append((base_res, model))
        if models is None:
            return None
        else:
            print("Finding the best model out of model types ran")
            best = compare_best_models(models)
            return best
