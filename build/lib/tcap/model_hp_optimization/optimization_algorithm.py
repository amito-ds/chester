from typing import List

from tcap.model_training.data_preparation import ComplexParameter
from tcap.model_training.models.model_input_and_output_classes import ModelInput
from .hp_space import sample_hp_space


class Optimizer:
    def __init__(self, f: callable, hp_space: List[ComplexParameter],
                 optimizer_type: str, model_input: ModelInput, model_type: str):
        self.f = f
        self.hp_space = hp_space
        self.optimizer_type = optimizer_type
        self.model_input = model_input
        self.model_type = model_type

    def random_search(self, num_iter: int):
        best_params = None
        best_score = float('-inf')
        for _ in range(num_iter):
            params = sample_hp_space(self.hp_space)
            model_input = ModelInput(cv_data=self.model_input.cv_data, parameters=params,
                                     target_col=self.model_input.target_col)
            score = self.f(model_input)
            if score > best_score:
                best_score = score
                best_params = params
        return best_params, best_score
