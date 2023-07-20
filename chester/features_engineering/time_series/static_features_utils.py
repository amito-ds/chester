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
        print("wow! col!", self.col_name)

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

#
# create a sample DataFrame with a datetime column
# data = pd.DataFrame({
#     "date": pd.date_range(start="2022-01-22 07:00:000", periods=10, freq="D"),
#     "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# })
#
# data = pd.DataFrame({
#     "date": pd.to_datetime(["2022-01-22 07:00:00", "2022-01-23 07:00:00", None, "2022-01-25 07:00:00", None, "2022-01-27 07:00:00", "2022-01-28 07:00:00", "2022-01-29 07:00:00", "2022-01-30 07:00:00", "2022-01-31 07:00:00"]),
#     "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# })
#
# # create a DataInfo object with the sample DataFrame
# data_info = DataInfo(data=data, target="value")
#
# # create a TSStaticFeatures object and call the run method to extract time-related features
# static_features = TSStaticFeatures(
#     column=data.date,
#     col_name="date",
#     time_series_handler=None,
#     data_info=data_info
# )
# features, feature_names = static_features.run()
#
# # print the extracted features and feature names
# print(features[0:3].T)
# print(feature_names)
