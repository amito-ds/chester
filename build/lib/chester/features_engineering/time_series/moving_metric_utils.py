import numpy as np
import pandas as pd

from chester.features_engineering.time_series.ts_utils import min_fix, max_fix, mean_fix, median_fix
from chester.run.user_classes import TimeSeriesHandler
from chester.zero_break.problem_specification import DataInfo


class MovingMetric:
    # the goal is to calculate moving metrics (like average or max) using last n lag values, for different n and metrics
    def __init__(self, column,
                 col_name,
                 time_series_handler: TimeSeriesHandler = None,
                 data_info: DataInfo = None):
        self.date_col_name = col_name  # the name of the date col
        # self.date_column = column
        self.time_series_handler = time_series_handler or TimeSeriesHandler()
        self.data_info = data_info
        self.df = self.data_info.data.sort_values(self.date_col_name)
        self.id_cols = self.time_series_handler.id_cols or []
        self.lag_values = time_series_handler.lag_values
        # self.target = self.df[self.data_info.target]

    def collect_last_values(self):
        if not self.id_cols:
            period_back, calculation_types = self.lag_values
            target = self.data_info.target
            max_lag = max(period_back)
            shifted_cols = [self.df[target].shift(lag + 1).rename(f'ts_lags_{lag + 1}') for lag in range(max_lag)]
            shifted_df = pd.concat(shifted_cols, axis=1)
            last_values = shifted_df.apply(lambda x: x.dropna().tolist(), axis=1).rename('last_values')
            self.df = pd.concat([self.df, last_values], axis=1)
        else:
            groups = self.id_cols
            period_back, calculation_types = self.lag_values
            target = self.data_info.target
            max_lag = max(period_back)
            shifted_cols = [self.df.groupby(groups)[target].shift(lag + 1).rename(f'ts_lags_{lag + 1}') for lag in
                            range(max_lag)]
            shifted_df = pd.concat(shifted_cols, axis=1)
            last_values = shifted_df.apply(lambda x: x.dropna().tolist(), axis=1).rename('last_values')
            self.df = pd.concat([self.df, last_values], axis=1)

    def calculate_ts_metrics(self):
        period_back, calculation_types = self.lag_values
        target = self.data_info.target
        date_col_name = self.date_col_name
        names = []
        for pb in period_back:
            # Take the first pb values from the last_values column
            values = self.df['last_values'].apply(lambda x: x[0:pb])

            for ct in calculation_types:
                if ct == 'min':
                    metric_values = values.apply(lambda x: min_fix(x))
                    name = f'ts_mm_{pb}_{ct}_{date_col_name}'
                    names.append(name)
                    self.df[name] = metric_values
                elif ct == 'max':
                    metric_values = values.apply(lambda x: max_fix(x))
                    name = f'ts_mm_{pb}_{ct}_{date_col_name}'
                    names.append(name)
                    self.df[name] = metric_values
                elif ct == 'mean':
                    metric_values = values.apply(lambda x: mean_fix(x))
                    name = f'ts_mm_{pb}_{ct}_{date_col_name}'
                    names.append(name)
                    self.df[name] = metric_values
                elif ct == 'median':
                    metric_values = values.apply(lambda x: median_fix(x))
                    name = f'ts_mm_{pb}_{ct}_{date_col_name}'
                    names.append(name)
                    self.df[name] = metric_values

        # Remove the last_values column from the DataFrame
        self.df.drop('last_values', axis=1, inplace=True)
        return names

    def run(self):
        period_back, calculation_types = self.lag_values
        self.collect_last_values()
        names = self.calculate_ts_metrics()
        return self.df, names

# df = pd.read_csv("/Users/amitosi/PycharmProjects/chester/chester/data/day.csv")
# df.rename(columns={'cnt': 'target'}, inplace=True)
# dat_info = DataInfo(data=df, target='target')
# dat_info.calculate()
# # print(dat_info)
#
# # ts_handler = TimeSeriesHandler()
# ts_handler = TimeSeriesHandler(id_cols=['workingday'])
# ma = MovingMetric(column=df['dteday'], col_name='dteday', data_info=dat_info, time_series_handler=ts_handler)
# ma.run()
# print(ma.df.columns)
