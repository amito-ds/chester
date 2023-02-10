import random

from chester.model_training.data_preparation import Parameter


class HPGenerator:
    def __init__(self,
                 best_practice_configs: list = None,
                 categorical_sample_configs: list = None,
                 n_models=9,
                 best_practice_prob=0.33):
        self.n_models = n_models
        self.best_practice_prob = best_practice_prob
        self.best_practice_configs = best_practice_configs
        self.categorical_sample_configs = categorical_sample_configs

    def generate_configs(self):
        configs = []
        for i in range(self.n_models):
            configs.append(self.generate_config())
        return configs

    def generate_config(self):
        if random.uniform(0, 1) < self.best_practice_prob:
            return self.generate_bp_config()
        else:
            return self.generate_random_config()

    def generate_bp_config(self):
        return random.choice(self.best_practice_configs)

    def generate_categorical_random_config(self):
        return random.choice(self.categorical_sample_configs)

    def generate_numerics_random_config(self) -> dict:
        pass

    def generate_random_config(self):
        categorical_config = self.generate_categorical_random_config()
        numerics_config = self.generate_numerics_random_config()
        combined_config = {**categorical_config, **numerics_config}
        return combined_config

    @staticmethod
    def hp_format(hp_list):

        hp_list_formatted = []

        for hp in hp_list:
            parameters = []
            for name, value in hp.items():
                parameters.append(Parameter(name, value))
            hp_list_formatted.append(parameters)
        return hp_list_formatted
