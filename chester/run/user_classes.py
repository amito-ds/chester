import warnings

from chester.run.feature_attention_utils import FeatureTypeCategorizer
from chester.zero_break.problem_specification import DataInfo

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
import logging

logging.basicConfig(level=logging.WARNING)
from chester.cleaning import cleaning_func as cln
from chester.preprocessing import preprocessing_func as pp

import pandas as pd


class Data:
    def __init__(self, df: pd.DataFrame, target_column: str = None):
        self.df = df
        self.target_column = target_column


class FeatureTypes:
    def __init__(self, feature_types: dict):
        self.feature_types = feature_types

    def fix_feature_types(self):
        self.feature_types = FeatureTypeCategorizer(self.feature_types).categorize_features()

    def update_data_info(self, data_info: DataInfo):
        self.fix_feature_types()
        data_info.feature_types_val = self.feature_types
        return data_info


class TextHandler:
    def __init__(self,
                 text_cleaner: cln.TextCleaner = None,
                 text_pre_process: pp.TextPreprocessor = None):
        self.text_cleaner = text_cleaner
        self.text_pre_process = text_pre_process


class TimeSeriesHandler:
    def __init__(self,
                 time_frequency=None,
                 n_series_features=3,
                 feature_types=None,
                 id_cols=None,
                 lag_values=None  # Dict of (events back, min/max/avg)
                 ):
        if feature_types is None:
            self.feature_types = ['static', 'freq', 'count', 'lag', 'cyclic']
        else:
            self.feature_types = feature_types
        self.time_frequency = time_frequency
        self.n_series_features = n_series_features
        self.id_cols = id_cols
        if lag_values is None:
            period_back = [1, 2, 3, 5, 7, 14, 21, 360, 1000]
            calc_type = ['mean', 'median', 'min', 'max']
            self.lag_values = period_back, calc_type
        else:
            self.lag_values = lag_values


class FeatureStats:
    def __init__(self, plot=True):
        self.plot = plot


class TextFeatureSpec:
    def __init__(self,
                 split_data: bool = True, split_prop: float = 0.3, split_random_state=42,
                 text_column="text", target_column='target',
                 corex=True, corex_dim=50, tfidf=True, tfidf_dim=100, bow=True, bow_dim=100,
                 ngram_range=(2, 1)):
        self.split_data = split_data
        self.split_prop = split_prop
        self.split_random_state = split_random_state
        self.text_column = text_column
        self.target_column = target_column
        self.corex = corex
        self.corex_dim = corex_dim
        self.tfidf = tfidf
        self.tfidf_dim = tfidf_dim
        self.bow = bow
        self.bow_dim = bow_dim
        self.ngram_range = ngram_range


class FeatureExtraction:
    def __init__(self, text_featrue_extraction: TextFeatureSpec):
        self.text_featrue_extraction = text_featrue_extraction


class PreModel:
    def __init__(self, plot=True):
        self.plot = plot


class ModelRun:
    def __init__(self, best_practice_prob=0.25, n_models=5):
        self.best_practice_prob = best_practice_prob
        self.n_models = n_models


class PostModel:
    def __init__(self, plot=True):
        self.plot = plot


class ModelWeakness:
    def __init__(self, plot=True):
        self.plot = plot
