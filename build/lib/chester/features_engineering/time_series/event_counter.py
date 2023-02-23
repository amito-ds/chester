import math

import numpy as np
import pandas as pd

from chester.features_engineering.time_series.ts_utils import min_fix, max_fix, mean_fix, median_fix
from chester.run.user_classes import TimeSeriesHandler
from chester.zero_break.problem_specification import DataInfo


class EventCounter:
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
        self.df[self.date_col_name] = pd.to_datetime(self.df[self.date_col_name])  # convert to datetime
        self.time_between_events = None

        self.time_between_events = None
        self.calculate_time_between_events()
        # self.target = self.df[self.data_info.target]

    def calculate_time_between_events(self):
        # Get the date column and id columns from self.df
        id_cols = self.id_cols

        if id_cols:
            self.df = self.df.sort_values(by=id_cols + [self.date_col_name], ascending=True)
            groups = self.df.groupby(id_cols)
            time_between_events = groups[self.date_col_name].diff().dt.total_seconds()
        else:
            date_col = self.df[self.date_col_name]
            date_col = date_col.sort_values(ascending=True)
            time_between_events = date_col.diff().dt.total_seconds()
        self.time_between_events = time_between_events
        self.df['time_between_events'] = time_between_events

    def collect_last_values(self):
        target = 'time_between_events'  # change to diff

        if not self.id_cols:
            period_back, calculation_types = self.lag_values
            max_lag = max(period_back)
            target = self.df[target]
            shifted_cols = [target.shift(lag + 1).rename(f'ts_lags_{lag + 1}') for lag in range(max_lag)]
            shifted_df = pd.concat(shifted_cols, axis=1)
            last_values = shifted_df.apply(lambda x: x.dropna().tolist(), axis=1).rename('last_values')
            self.df = pd.concat([self.df, last_values], axis=1)
        else:
            groups = self.id_cols
            period_back, calculation_types = self.lag_values
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
        suffix = "ts_freq"

        measurement, t = self.time_series_handler.time_frequency
        # Calculate the translated value of t
        translated_t = t
        if measurement in ("second", "seconds"):
            translated_t = t
        elif measurement in ("minute", "minutes"):
            translated_t = t * 60
        elif measurement in ("hour", "hours"):
            translated_t = t * 3600
        elif measurement in ("day", "days"):
            translated_t = t * 86400
        elif measurement in ("week", "weeks"):
            translated_t = t * 604800
        elif measurement in ("month", "months"):
            translated_t = t * 2592000
        elif measurement in ("year", "year"):
            translated_t = t * 31536000
        else:
            pass

        for pb in period_back:
            # Take the first pb values from the last_values column
            total_times = self.df['last_values']. \
                apply(lambda x: np.sum(x[0:pb])). \
                apply(lambda x: math.floor(x / translated_t))
            print(total_times)

    def run(self):
        self.collect_last_values()
        self.calculate_ts_metrics()
        return self.df
