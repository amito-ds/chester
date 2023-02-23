import math

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from chester.run.user_classes import TimeSeriesHandler
from chester.util import remove_prefix_suffix
from chester.zero_break.problem_specification import DataInfo


class TimeSeriesFeatureStatistics:
    def __init__(self,
                 data_info: DataInfo,
                 time_series_handler: TimeSeriesHandler,
                 max_print=None,
                 ts_cols=None):
        self.data_info = data_info
        self.max_print = max_print
        self.time_series_handler = time_series_handler
        self.time_cols = list(self.data_info.feature_types_val["time"])
        self.ts_cols = ts_cols
        self.data = self.data_info.data[list(set(list(self.ts_cols) + list(self.time_cols)))]
        self.data = self.data.sample(min(10000, len(self.data)))  # sample
        self.time_frequency = time_series_handler.time_frequency

    @staticmethod
    def discover_scale(column):
        median = column.median()
        if median >= 31536000:
            return "years", 31536000
        elif median > 2592000:
            return "months", 2592000
        elif median > 604800:
            return "weeks", 604800
        elif median > 86400:
            return "days", 86400
        elif median >= 3600:
            return "hours", 3600
        elif median >= 60:
            return "minutes", 60
        else:
            return "seconds", 1

    def plot_time_between_events(self, time_col, plot=False):
        # all_columns = self.data_info.get_all_features()
        all_columns = [name for name in self.data_info.data.columns if name.startswith("ts_")]

        _, metric_options = self.time_series_handler.lag_values
        freq_cols = [name for name in all_columns if
                     name.startswith("ts_freq_") and name.endswith(time_col) and metric_options[0] in name]
        freq_cols.sort()
        dim_plots = math.floor(math.sqrt(len(freq_cols)))  # plot dim*dim graphs, on the same plot
        if dim_plots == 0:
            plt.close()
            return None
        # Create a subplot grid for the histograms
        fig, axs = plt.subplots(dim_plots, dim_plots, figsize=(10, 7))

        # Plot a histogram for each column in freq_cols and add a title
        for i, col in enumerate(freq_cols):
            if i > dim_plots * dim_plots - 1:
                break
            row_index = i // dim_plots
            col_index = i % dim_plots

            clean_col = remove_prefix_suffix(string=col, prefix="ts_freq_", suffix=f"_{time_col}")
            scale, scaler = self.discover_scale(self.data[col])
            axs[row_index, col_index].hist(self.data[col] / scaler, bins=50)
            axs[row_index, col_index].set_title(f"{clean_col} ({scale})")

        # Add a main title and adjust the subplot layout
        fig.suptitle(f"Distribution of Time Between Last (index=lag) Events ({time_col.title()})")
        fig.tight_layout()

        # Show the plot
        if plot:
            plt.show()
        plt.close()

    def plot_dates(self, time_col, time_freq, plot=True):
        # Get the time frequency and convert it to a pandas frequency string
        time_freq = time_freq[0].lower()
        if time_freq.endswith("s"):
            time_freq = time_freq[:-1]
        freq_str = \
            {"year": "%Y-01-01", "month": "%Y-%m-01", "week": "%Y-%U-0", "day": "%Y-%m-%d", "hour": "%Y-%m-%d %H",
             "minute": "%Y-%m-%d %H:%M", "second": "%Y-%m-%d %H:%M:%S"}[time_freq]

        # Set the figure size and style
        plt.figure(figsize=(9, 7))
        plt.style.use("seaborn-whitegrid")

        # Convert the time column to a pandas datetime column and format it to the requested time frequency
        column = pd.to_datetime(self.data[time_col]).dt.strftime(freq_str)

        # Count the number of occurrences of each date and sort the dates in ascending order
        counts = column.value_counts().sort_index()

        # Plot the data with appropriate title and x-axis label
        plt.plot(counts.index, counts.values, "o-")
        plt.title("Number of occurrences by {}".format(time_freq))
        plt.xlabel(time_freq.capitalize())
        plt.ylabel("Count")

        # Set the x-axis tick locations and labels to show a sample of the dates
        sample_size = min(20, len(counts))
        xticks = range(0, len(counts), len(counts) // sample_size)
        xtick_labels = [counts.index[i] for i in xticks]
        plt.xticks(xticks, xtick_labels, rotation=45)

        # Show the plot
        if plot:
            plt.show()
        plt.close()

    def plot_date_parts(self, time_col, plot=False):
        # Get the columns that represent date parts of the time column
        all_columns = self.data_info.get_all_features()
        optional_col_prefix = [
            'ts_dayinmonth_', 'ts_dayinweek_', 'ts_month_', 'ts_quarter_', 'ts_year', 'ts_minutes', 'ts_hours']
        date_part_cols = [name for name in all_columns if
                          any(name.startswith(prefix) for prefix in optional_col_prefix) and name.endswith(time_col)]

        # Create a subplot for each date part column and plot a heatmap of the value counts
        num_plots = len(date_part_cols)
        fig, axs = plt.subplots(num_plots, 1, figsize=(14, 3 * num_plots), sharey=False)
        for i, col in enumerate(date_part_cols):
            clean_col = remove_prefix_suffix(string=col, prefix="ts_", suffix=f"_{time_col}")

            # Count the number of occurrences of each unique value in the column
            counts = self.data[col].value_counts(normalize=True)[:14]

            # Get the unique values and their corresponding percentages
            values = counts.index

            # Sort the values in the order of the value labels
            values_sorted = sorted(values)

            # Reorder the corresponding value counts and percentages accordingly
            counts_sorted = counts.reindex(values_sorted)
            percentages_sorted = np.round(counts_sorted.values * 100).astype(int)

            # Create a matrix of percentages with one row and the same number of columns as unique values
            percent_matrix = np.array([percentages_sorted])

            # Plot the heatmap with appropriate title and axis labels
            sns.heatmap(percent_matrix, ax=axs[i], cmap="Blues", annot=True, fmt=".0f", cbar=False)

            # Set the x-axis tick labels to the unique values of the feature in the order of the value labels
            axs[i].set_xticklabels(values_sorted)

            axs[i].set_ylabel(clean_col)

        # Set the overall x-axis label and adjust the subplot layout
        fig.suptitle(f"Percentage of Occurrences by {time_col} Column Parts:")
        axs[-1].set_xlabel("Values")
        fig.tight_layout()

        # Show the plot
        if plot:
            plt.show()
        plt.close()

    def run_single(self, time_col, plot=True):
        self.plot_dates(time_col=time_col, time_freq=self.time_frequency, plot=plot)

    def run(self, plot=True):
        for time_col in self.time_cols:
            self.run_single(time_col, plot)
            self.plot_date_parts(time_col, plot)
            self.plot_time_between_events(time_col, plot)
