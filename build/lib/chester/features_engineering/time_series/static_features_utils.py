import pandas as pd

from chester.run.user_classes import TimeSeriesHandler
from chester.zero_break.problem_specification import DataInfo


class TSStaticFeatures:
    def __init__(self, column,
                 col_name,
                 time_series_handler: TimeSeriesHandler = None,
                 data_info: DataInfo = None):
        self.col_name = col_name
        self.column = column
        self.time_series_handler = time_series_handler
        self.data_info = data_info
        self.df = self.data_info.data
        self.df[self.col_name] = pd.to_datetime(self.df[self.col_name])  # conver to datetime

    def extract_minute(self):
        """
        Extract the minute in the day from a datetime column.
        :param df: pandas DataFrame
        :param datetime_col: string, the name of the datetime column
        :return: pandas series
        """
        return self.df[self.col_name].dt.hour * 60 + self.df[self.col_name].dt.minute, "ts_minutes_" + self.col_name

    def extract_hour(self):
        """
        Extract the hour from the datetime column.
        :return: pandas series
        """
        return self.df[self.col_name].dt.hour, "ts_hours_" + self.col_name

    def extract_day_in_month(self):
        """
        Extract the day of the month from the datetime column.
        :return: pandas series
        """
        return self.df[self.col_name].dt.day, "ts_dayinmonth_" + self.col_name

    def extract_day_in_week(self):
        """
        Extract the day of the week from the datetime column.
        :return: pandas series
        """
        return self.df[self.col_name].dt.strftime("%A"), "ts_dayinweek_" + self.col_name

    def extract_month_in_year(self):
        """
        Extract the month of the year from the datetime column.
        :return: pandas series
        """
        return self.df[self.col_name].dt.month, "ts_month_" + self.col_name

    def extract_quarter(self):
        """
        Extract the quarter of the year from the datetime column.
        :return: pandas series
        """
        return self.df[self.col_name].dt.quarter, "ts_quarter_" + self.col_name

    def extract_year(self):
        """
        Extract the year from the datetime column.
        :return: pandas series
        """
        return self.df[self.col_name].dt.year, "ts_year_" + self.col_name

    def run(self):
        # call all feature extraction methods and concatenate the resulting Pandas series
        minute, minute_name = self.extract_minute()
        hour, hour_name = self.extract_hour()
        day, day_name = self.extract_day_in_month()
        weekday, weekday_name = self.extract_day_in_week()
        month, month_name = self.extract_month_in_year()
        quarter, quarter_name = self.extract_quarter()
        year, year_name = self.extract_year()
        ts_feat_names = [minute_name, hour_name, day_name, weekday_name, month_name, quarter_name, year_name]
        ts_features = pd.concat([minute, hour, day, weekday, month, quarter, year], axis=1)
        ts_features.columns = ts_feat_names
        return ts_features, ts_feat_names
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
