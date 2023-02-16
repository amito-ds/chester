import pandas as pd

from chester.features_engineering.time_series.cyclic_features import CyclicFeatures
from chester.features_engineering.time_series.feature_elimination_utils import FeatureEliminationUtils
from chester.features_engineering.time_series.get_time_freqeuency_utils import TimeFrequencyDecider
from chester.features_engineering.time_series.moving_average_utils import MovingMetric
from chester.features_engineering.time_series.static_features_utils import TSStaticFeatures
from chester.run.user_classes import TimeSeriesHandler
from chester.zero_break.problem_specification import DataInfo


class TimeSeriesFeatureExtraction:
    # after getting the data
    # eliminate
    # update the original data and in data info
    # update the feature types in data info
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
        self.time_frequency = time_series_handler.time_frequency or self._get_time_frequency()

    def static_features(self):
        stat_features = TSStaticFeatures(column=self.column,
                                         col_name=self.col_name,
                                         time_series_handler=self.time_series_handler,
                                         data_info=self.data_info)
        df, names = stat_features.run()
        df = FeatureEliminationUtils(df=df[names]).run()  # Elimination for the relevant features only
        self.data_info.data = pd.concat([self.data_info.data, df], axis=1)
        # update data info feature types

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
        # Eliminate features
        self.data_info.data = mm.run()  # update the table
        # update the feature types

    # time to last event, time to last 2 events (avg of not nulls, except the first event)
    def freq_features(self):
        pass

    # use T to count events in the last T*(1, 2, 3, 5, 10, 20)
    def count_events_features(self):
        t = self.time_frequency
        pass

    # sin, cos, on the static_features
    def cyclic_features(self):
        cf = CyclicFeatures(column=self.column,
                            col_name=self.col_name,
                            time_series_handler=self.time_series_handler,
                            data_info=self.data_info)
        df, names = cf.run()
        df = FeatureEliminationUtils(df=df[names]).run()  # Elimination for the relevant features only
        self.data_info.data = pd.concat([self.data_info.data, df], axis=1)
        # update data info feature types

    def run(self):
        self.static_features()  # return df, feature names
        self.cyclic_features()


df = pd.read_csv("/Users/amitosi/PycharmProjects/chester/chester/data/day.csv")
df.rename(columns={'cnt': 'target'}, inplace=True)
dat_info = DataInfo(data=df, target='target')
dat_info.calculate()
# print(dat_info)

# ts_handler = TimeSeriesHandler()
ts_handler = TimeSeriesHandler(id_cols=['workingday'])
tsfe = TimeSeriesFeatureExtraction(column=df['dteday'], col_name='dteday', data_info=dat_info,
                                   time_series_handler=ts_handler)
tsfe.run()
