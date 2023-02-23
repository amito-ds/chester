import math

import matplotlib.pyplot as plt
import pandas as pd

from chester.run.user_classes import TimeSeriesHandler
from chester.zero_break.problem_specification import DataInfo
import seaborn as sns


class TimeSeriesPreModelAnalysis:
    def __init__(self, data_info: DataInfo, time_series_handler: TimeSeriesHandler):
        self.data_info = data_info
        self.time_series_handler = time_series_handler
        self.target_col = data_info.target
        self.target = self.data_info.data[self.target_col]
        self.date_cols = self.data_info.feature_types_val["time"]
        relevant_cols = [self.target_col] + \
                        list(self.data_info.feature_types_val["time"]) + \
                        [name for name in self.data_info.data.columns if name.startswith("ts_")]
        self.data = self.data_info.data[relevant_cols]

    def analyze_single_moving_metric(self, date_col):
        ts_cols = [name for name in self.data_info.data.columns if name.startswith("ts_")]
        moving_metric_str = "ts_mm_"
        _, metric_options = self.time_series_handler.lag_values

        relevant_cols = [name for name in ts_cols if
                         moving_metric_str in name and name.endswith(date_col) and metric_options[0] in name]
        df = self.data[[self.target_col] + list(set(relevant_cols))]
        if len(relevant_cols) > 0:
            self.plot_correlation_and_partial_dependence(df)
        plt.show()

    @staticmethod
    def plot_correlation_and_partial_dependence(df):
        # Separate the target column and the moving metric columns
        target_col = df.columns[0]
        moving_metric_cols = df.columns[1:]

        # Calculate the number of rows and columns for the partial dependence grid
        k = math.floor(math.sqrt(len(moving_metric_cols)))
        num_rows = num_cols = k

        # Create a figure with two subplots
        fig, ax = plt.subplots(figsize=(12, 4))

        # Plot the correlation heatmap in the top subplot
        corr_matrix = df.corr()[target_col][1:]
        sns.heatmap(pd.DataFrame(corr_matrix).transpose(), cmap="coolwarm", vmin=-1, vmax=1, annot=True, ax=ax)
        ax.set_title("Correlation between features and target")
        ax.set_xlabel("")
        plt.show()
        plt.close()

        # Plot the partial dependence grid in the bottom subplot
        # Create a figure with two subplots
        fig, ax = plt.subplots(figsize=(14, 4 + k * 4))
        for i, moving_metric_col in enumerate(moving_metric_cols):
            if i >= k * k:
                break
            row = i // num_cols
            col = i % num_cols
            ax = fig.add_subplot(num_rows, num_cols, i+1)
            ax.plot(df[moving_metric_col], df[target_col], '.', alpha=0.1)
            ax.set_xlabel(moving_metric_col)
            ax.set_ylabel(target_col)
            if col != 0:
                ax.set_yticklabels([])
            if row != num_rows - 1:
                ax.set_xticklabels([])

        fig.suptitle("Partial Dependence Plots Between Target and Moving Average of the Past Periods")
        # fig.subplots_adjust(wspace=0.2, hspace=0.5)
        plt.show()
        plt.close()

    def analyze_moving_metric(self):
        for date_col in self.date_cols:
            self.analyze_single_moving_metric(date_col=date_col)

    def run(self):
        self.analyze_moving_metric()
