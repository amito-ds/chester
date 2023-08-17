import pandas as pd

from chester.run.user_classes import TimeSeriesHandler
from chester.zero_break.problem_specification import DataInfo
from chester.run.user_classes import TimeSeriesHandler
from chester.zero_break.problem_specification import DataInfo


class TSStaticFeatures:
    def __init__(self, column, col_name, time_series_handler: TimeSeriesHandler = None, data_info: DataInfo = None):
        self.col_name = col_name
        self.column = column
        self.time_series_handler = time_series_handler
        self.data_info = data_info
        self.df = self.data_info.data
        try:
            self.df[self.col_name] = pd.to_datetime(self.df[self.col_name])  # convert to datetime
            self.is_datetime = True
        except Exception:
            self.is_datetime = False

    def safe_extract(self, method, col_name):
        try:
            return method(self.df[self.col_name]), f"ts_{col_name}_{self.col_name}"
        except:
            return pd.Series(0, index=self.df.index), f"ts_{col_name}_{self.col_name}"

    def extract_minute(self):
        return self.safe_extract(lambda col: col.dt.hour * 60 + col.dt.minute, 'minutes')

    def extract_hour(self):
        return self.safe_extract(lambda col: col.dt.hour, 'hours')

    def extract_day_in_month(self):
        return self.safe_extract(lambda col: col.dt.day, 'dayinmonth')

    def extract_day_in_week(self):
        return self.safe_extract(lambda col: col.dt.strftime("%A"), 'dayinweek')

    def extract_month_in_year(self):
        return self.safe_extract(lambda col: col.dt.month, 'month')

    def extract_quarter(self):
        return self.safe_extract(lambda col: col.dt.quarter, 'quarter')

    def extract_year(self):
        return self.safe_extract(lambda col: col.dt.year, 'year')

    def run(self):
        # call all feature extraction methods and concatenate the resulting Pandas series
        methods = [self.extract_minute, self.extract_hour, self.extract_day_in_month,
                   self.extract_day_in_week, self.extract_month_in_year, self.extract_quarter, self.extract_year]

        ts_features = pd.concat([method()[0] for method in methods], axis=1)
        ts_features.columns = [method()[1] for method in methods]
        return ts_features, ts_features.columns
