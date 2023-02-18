from chester.run.user_classes import TimeSeriesHandler
from chester.zero_break.problem_specification import DataInfo

import pandas as pd
from chester.run.user_classes import TimeSeriesHandler
from chester.zero_break.problem_specification import DataInfo
import random


class TimeFrequencyDecider:
    def __init__(self, column,
                 col_name,
                 time_series_handler: TimeSeriesHandler = None,
                 data_info: DataInfo = None):
        self.date_col_name = col_name  # the name of the date col
        self.column = column
        self.time_series_handler = time_series_handler or TimeSeriesHandler()
        self.data_info = data_info
        self.df = self.data_info.data
        self.id_cols = self.time_series_handler.id_cols or []
        self.df[self.date_col_name] = pd.to_datetime(self.df[self.date_col_name])  # convert to datetime
        self.time_between_events = None
        self.calculate_time_between_events()

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

        # Filter out any rows with null values in the new column
        self.time_between_events = time_between_events[~time_between_events.isna()]
        return time_between_events[~time_between_events.isna()]

    def _calculate_raw_frequency(self):
        time_between_events = self.time_between_events
        n = len(time_between_events)
        k = int(0.3 * n)  # Number of values to drop
        sorted_x = sorted(time_between_events)
        remaining_values = sorted_x[:-k]  # Drop top k values
        avg = sum(remaining_values) / len(remaining_values)
        return avg

    @staticmethod
    def translate(t):
        intervals = [
            (1, "second"),
            (15, "seconds"),
            (30, "seconds"),
            (60, "minute"),
            (180, "minutes"),
            (300, "minutes"),
            (900, "minutes"),
            (1800, "minutes"),
            (3600, "hour"),
            (10800, "hours"),
            (21600, "hours"),
            (43200, "hours"),
            (86400, "day"),
            (259200, "days"),
            (604800, "week"),
            (1209600, "weeks"),
            (1814400, "weeks"),
            (2419200, "weeks"),
            (2678400, "month"),
            (7776000, "months"),
            (15552000, "months"),
            (31536000, "year"),
            (63072000, "years"),
            (94608000, "years")
        ]

        # Find the first interval with a value greater than t
        for value, measurement in intervals:
            if t < value:
                t = value
                break

        # Calculate the translated value of t
        # ["second", "seconds","minute", "minutes" ,"hour", "hours","day", "days", "week",
        # "weeks","month", "months", "year", "years"]
        if measurement in ("second", "seconds"):
            translated_t = t
        elif measurement in ("minute", "minutes"):
            translated_t = t / 60
        elif measurement in ("hour", "hours"):
            translated_t = t / 3600
        elif measurement in ("day", "days"):
            translated_t = t / 86400
        elif measurement in ("week", "weeks"):
            translated_t = t / 604800
        elif measurement in ("month", "months"):
            translated_t = t / 2592000
        else:
            translated_t = t / 31536000

        return measurement, translated_t

    def run(self):
        raw_freq = self._calculate_raw_frequency()
        return self.translate(raw_freq)

# dates = ["2022-02-15 " + str(i).zfill(2) + ":00:00" for i in range(24)] * 1

# import random
#
# dates = []
# for _ in range(100):
#     # Choose a random hour, minute, and second
#     hour = random.randint(0, 23)
#     minute = random.randint(0, 59)
#     second = random.randint(0, 59)
#
#     # Construct a string in ISO format with the chosen time
#     date_str = f"2022-02-15 {hour:02d}:{minute:02d}:{second:02d}"
#
#     # Add the string to the list of dates
#     dates.append(date_str)
#
# dates = [date if random.random() > 0.1 else None for date in dates]
# # Create a DataFrame with the dates
# data = pd.DataFrame({
#     "date": pd.to_datetime(dates),
#     "value": 1  # The value is not important and can be flat
# })
# data = data.assign(ID=["A"] * (len(data) // 1) + ["A"][:len(data) % 1])
#
# # Create a DataInfo object with the sample DataFrame
# data_info = DataInfo(data=data, target="value")
#
# # Create a TimeSeriesHandler object with the date column
# decider = TimeFrequencyDecider(
#     column=data.date,
#     col_name="date",
#     time_series_handler=TimeSeriesHandler(id_cols=["ID"]),
#     data_info=data_info
# )
#
# df = decider.calculate_time_between_events()
# # print(df)
# # Print the resulting DataFrame with the time between events
# print(decider.run())
