from chester.features_engineering.time_series.get_time_freqeuency_utils import TimeFrequencyDecider
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
        self.time_series_handler = time_series_handler
        self.data_info = data_info
        self.time_frequency = \
            time_series_handler.time_frequency if time_series_handler.time_frequency is not None \
                else self.get_time_frequency()

    # day in month, hour of the day, year, ...

    def static_features(self):
        stat_features = TSStaticFeatures(column=self.column,
                                         col_name=self.col_name,
                                         time_series_handler=self.time_series_handler,
                                         data_info=self.data_info)
        return stat_features.run()

    # return (interval, counts)
    def _get_time_frequency(self):
        decider = TimeFrequencyDecider(column=self.column,
                                       col_name=self.col_name,
                                       time_series_handler=self.time_series_handler,
                                       data_info=self.data_info)
        return decider.run()

    def target_lags(self):
        pass

    def freq_features(self):
        pass

    def count_events_features(self):
        pass

    def cyclic_features(self):
        pass

    def run(self):
        pass  # return df, feature names
