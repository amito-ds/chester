import warnings

from chester.chapter_messages import chapter_message
from chester.model_analyzer.model_analysis import analyze_model
from chester.text_stats_analysis.data_quality import TextAnalyzer
from chester.text_stats_analysis.smart_text_analyzer import analyze_text_df
from chester.zero_break.problem_specification import DataInfo

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
import logging

logging.basicConfig(level=logging.WARNING)

from chester.model_training.data_preparation import CVData
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from chester.cleaning import cleaning_func as cln
from chester.preprocessing import preprocessing_func as pp
from chester.features_engineering import fe_nlp as fe_main
from chester.feature_analyzing import feature_correlation
from chester.model_training import data_preparation
from chester.model_training import best_model as bm
from chester.model_training.model_utils import analyze_results


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
