import warnings

from chester.zero_break.problem_specification import DataInfo

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
import logging

logging.basicConfig(level=logging.WARNING)

import pandas as pd


class Data:
    def __init__(self, df: pd.DataFrame, target_column: str = None):
        self.df = df
        self.target_column = target_column


class FeatureTypes:
    def __init__(self, feature_types: dict):
        self.feature_types = feature_types

    def update_data_info(self, data_info: DataInfo):
        data_info.feature_types_val = self.feature_types
        return data_info
        # FeatureTypes : {'numeric': [], 'boolean': [], 'text': [], 'categorical': [], 'time': []}


class TextHandler:
    def __init__(self):
        pass


class TextAnalyze:
    def __init__(self):
        pass


class FeatureStats:
    def __init__(self):
        pass


class PreModel:
    def __init__(self):
        pass


class ModelRun:
    def __init__(self):
        pass


class PostModel:
    def __init__(self):
        pass


class ModelWeakness:
    def __init__(self):
        pass
