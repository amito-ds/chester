import pandas as pd

from chester.features_engineering.time_series.cyclic_features_utils import CyclicFeatures
from chester.features_engineering.time_series.feature_elimination_utils import FeatureEliminationUtils
from chester.features_engineering.time_series.frequencies_utils import FrequenciesFeatures
from chester.features_engineering.time_series.get_time_freqeuency_utils import TimeFrequencyDecider
from chester.features_engineering.time_series.moving_metric_utils import MovingMetric
from chester.features_engineering.time_series.static_features_utils import TSStaticFeatures
from chester.run.user_classes import TimeSeriesHandler
from chester.zero_break.problem_specification import DataInfo


class TimeSeriesFeatureExtraction:
    def __init__(self,
                 column,
                 col_name,
                 time_series_handler: TimeSeriesHandler = None,
                 data_info: DataInfo = None
                 ):
        self.col_name = col_name
        self.column = column
        self.time_series_handler = time_series_handler or TimeSeriesHandler()
        self.data_info = data_info
        # save
        self.time_frequency = self.time_series_handler.time_frequency or self._get_time_frequency()
        self.time_series_handler.time_frequency = self.time_frequency

    def static_features(self):
        stat_features = TSStaticFeatures(column=self.column,
                                         col_name=self.col_name,
                                         time_series_handler=self.time_series_handler,
                                         data_info=self.data_info)
        sf_df, names = stat_features.run()
        sf_df = FeatureEliminationUtils(df=sf_df[names]).run()  # Elimination for the relevant features only
        self.data_info.data = pd.concat([self.data_info.data, sf_df], axis=1)  # update data
        self.data_info.feature_types_val["categorical"] = list(
            self.data_info.feature_types_val["categorical"])  # convert set to list
        self.data_info.feature_types_val["categorical"].extend(list(sf_df.columns))  # update features

    # return (interval, counts)
    def _get_time_frequency(self):
        decider = TimeFrequencyDecider(column=self.column,
                                       col_name=self.col_name,
                                       time_series_handler=self.time_series_handler,
                                       data_info=self.data_info)
        return decider.run()

    def target_lags(self):
        mm = MovingMetric(column=self.column,
                          col_name=self.col_name,
                          time_series_handler=self.time_series_handler,
                          data_info=self.data_info)
        target_df, names = mm.run()
        target_df = FeatureEliminationUtils(df=target_df[names]).run()  # Elimination for the relevant features only
        self.data_info.data = pd.concat([self.data_info.data, target_df], axis=1)  # update data
        self.data_info.feature_types_val["numeric"] = list(
            self.data_info.feature_types_val["numeric"])  # convert set to list
        self.data_info.feature_types_val["numeric"].extend(target_df.columns)  # update features

    def freq_features(self):
        ff = FrequenciesFeatures(column=self.column,
                                 col_name=self.col_name,
                                 time_series_handler=self.time_series_handler,
                                 data_info=self.data_info)
        df_ff, names = ff.run()
        df_ff = FeatureEliminationUtils(df=df_ff[names]).run()  # Elimination for the relevant features only
        self.data_info.data = pd.concat([self.data_info.data, df_ff], axis=1)  # update data
        self.data_info.feature_types_val["numeric"] = list(
            self.data_info.feature_types_val["numeric"])  # convert set to list
        self.data_info.feature_types_val["numeric"].extend(df_ff.columns)  # update features

    def cyclic_features(self):
        cf = CyclicFeatures(column=self.column,
                            col_name=self.col_name,
                            time_series_handler=self.time_series_handler,
                            data_info=self.data_info)
        df_cf, names = cf.run()
        df_cf = FeatureEliminationUtils(df=df_cf[names]).run()  # Elimination for the relevant features only
        self.data_info.data = pd.concat([self.data_info.data, df_cf], axis=1)
        self.data_info.feature_types_val["numeric"] = list(
            self.data_info.feature_types_val["numeric"])  # convert set to list
        self.data_info.feature_types_val["numeric"].extend(df_cf.columns)  # update features

    def run(self):
        self.static_features()  # return df, feature names
        self.cyclic_features()
        if 'regression' in self.data_info.problem_type_val.lower():
            self.target_lags()
        self.freq_features()
