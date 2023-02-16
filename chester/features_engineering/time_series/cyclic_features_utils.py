import numpy as np
import pandas as pd

from chester.run.user_classes import TimeSeriesHandler
from chester.zero_break.problem_specification import DataInfo


class CyclicFeatures:
    def __init__(self, column,
                 col_name,
                 time_series_handler: TimeSeriesHandler = None,
                 data_info: DataInfo = None):
        self.date_col_name = col_name
        self.column = column
        self.time_series_handler = time_series_handler
        self.data_info = data_info
        self.df = self.data_info.data

    def extract_relevant_date_cols(self):
        optional_static_features = ['ts_minutes', 'ts_hours', 'ts_dayinmonth', 'ts_month',
                                    'ts_quarter', 'ts_year']
        all_features = self.df.columns
        relevant_features = [feat for feat in all_features if feat.startswith(tuple(optional_static_features))]
        date_cols = [col for col in relevant_features if self.date_col_name in col]
        return date_cols

    def calculate_cyclic_features(self):
        cols = self.extract_relevant_date_cols()
        out_cols = []
        for col in cols:
            # Calculate the sine and cosine of the date column
            sin_col_name = f"{col}_sin"
            cos_col_name = f"{col}_cos"
            sin_values = self.df[col].apply(lambda x: np.sin(2 * np.pi * x / self.df[col].max()))
            cos_values = self.df[col].apply(lambda x: np.cos(2 * np.pi * x / self.df[col].max()))

            # Add the new columns to the DataFrame
            self.df[sin_col_name] = sin_values
            self.df[cos_col_name] = cos_values
            out_cols.append(sin_col_name)
            out_cols.append(cos_col_name)
        return out_cols

    def run(self):
        names = self.calculate_cyclic_features()
        return self.df, names
