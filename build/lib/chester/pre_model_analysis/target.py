import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.errors import SettingWithCopyWarning

from chester.feature_stats.categorical_stats import CategoricalStats
from chester.feature_stats.numeric_stats import NumericStats
from chester.run.user_classes import TimeSeriesHandler
from chester.util import remove_prefix_suffix
from chester.zero_break.problem_specification import DataInfo


class TargetPreModelAnalysis:
    def __init__(self, data_info: DataInfo, time_series_handler: TimeSeriesHandler):
        import warnings
        warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
        self.data_info = data_info
        self.time_series_handler = time_series_handler
        self.target_col = self.data_info.target
        self.target = self.data_info.data[self.target_col]
        relevant_cols = [self.target_col] + \
                        list(self.data_info.feature_types_val["time"]) + \
                        [name for name in self.data_info.data.columns if name.startswith("ts_")]
        relevant_cols = list(set(relevant_cols))
        self.target_df = self.data_info.data[relevant_cols]  # select target, date and ts cols
        self.target_df.rename(columns={self.data_info.target: 'target_label'}, inplace=True)
        self.target_data_info = DataInfo(data=self.target_df.sample(min(len(self.target_df), 10000)))
        self.target_data_info.calculate()
        self.target_type = None
        # target type
        for feature_type, features in self.target_data_info.feature_types_val.items():
            if 'target_label' in features:
                self.target_type = feature_type

    def plot_histogram(self):
        target = self.target
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.hist(target, bins=30, edgecolor='k', alpha=0.5, label='Histogram')
        ax.legend(loc='upper left')
        ax.set_xlabel('Values')
        ax.set_ylabel('Counts')
        ax.set_title(f'Histogram of {self.target.name}')
        ax2 = ax.twinx()
        sns.kdeplot(target, ax=ax2, label='Density')
        ax2.legend(loc='upper right')
        ax2.set_ylabel('Density')
        plt.show()
        plt.close()

    def plot_barplot(self):
        from chester.util import ReportCollector, REPORT_PATH
        rc = ReportCollector(REPORT_PATH)

        target = self.target
        value_counts = target.value_counts()

        # df_to_save = value_counts.to_dict(orient='records')[0]
        rc.save_object(obj=value_counts[0:10], text="values counts for categorical target to predict:")

        percentages = np.array(value_counts / target.size * 100)
        fig, ax1 = plt.subplots(figsize=(9, 9), dpi=100)
        ax2 = ax1.twinx()

        # Create a color map that shows the percentage of each target value
        cmap = sns.light_palette("green", as_cmap=True)
        heatmap = np.array([percentages, ] * len(value_counts))

        # Create a bar plot with a heatmap color scheme
        ax1.barh(value_counts.index, value_counts.values)
        ax1.set_xlabel('Counts')
        ax1.set_ylabel('Values')
        ax1.invert_yaxis()

        # Set the y-axis tick labels to the target values
        # ax1.set_yticklabels(value_counts.index)

        # Add a colorbar for the heatmap
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=percentages.min(), vmax=percentages.max()))
        sm._A = []
        cbar = fig.colorbar(sm, ax=ax1)
        cbar.ax.set_ylabel('Percentages')

        ax2.set_ylim([0, 100])
        ax2.plot(value_counts.index, percentages, color='red', marker='o')
        ax2.set_ylabel('Percentages', color='red')
        ax1.set_title(f'Bar Plot of {self.target.name}')
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
        df = self.target_data_info.data.copy()
        df[time_col] = pd.to_datetime(df[time_col], format=freq_str)
        df = df.sort_values(time_col)

        # Count the number of occurrences of each date and sort the dates in ascending order
        counts = df[time_col].value_counts().sort_index()

        # Plot the data with appropriate title and x-axis label
        ax = plt.gca()
        ax.plot(counts.index, counts.values, "o-", label='Count')
        ax.set_xlabel(time_freq.capitalize())
        ax.set_ylabel("Count")

        # Set the x-axis tick locations and labels to show a sample of the dates
        sample_size = min(10, len(counts))
        xticks = range(0, len(counts), len(counts) // sample_size)
        xtick_labels = [counts.index[i].strftime(freq_str) for i in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, rotation=45)

        # Get the average of the target column and plot it on a secondary y-axis
        target_mean = self.target.groupby(df[time_col]).mean()
        ax2 = ax.twinx()
        ax2.plot(target_mean.index, target_mean.values, "o-", color="C1", label='Mean')
        ax2.set_ylabel(self.target_col, color="C1")
        ax2.tick_params(axis='y', labelcolor="C1")

        # Show the legend
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='best')

        # Set the title
        ax.set_title("Number of occurrences by {} and mean {}".format(time_freq, self.target_col))

        # Show the plot
        if plot:
            plt.show()
        plt.close()

    def plot_date_parts(self, time_col, plot=False):
        # Get the columns that represent date parts of the time column
        df = self.target_data_info.data.copy()
        all_columns = self.data_info.get_all_features()
        optional_col_prefix = [
            'ts_dayinmonth_', 'ts_dayinweek_', 'ts_month_', 'ts_quarter_', 'ts_year', 'ts_minutes', 'ts_hours']
        date_part_cols = [name for name in all_columns if
                          any(name.startswith(prefix) for prefix in optional_col_prefix) and name.endswith(time_col)]

        # Create a subplot for each date part column and plot two rows: percentage of
        # occurrences and mean target variable
        num_plots = len(date_part_cols)
        fig, axs = plt.subplots(num_plots * 2, 1, figsize=(14, 3 * num_plots), sharey=False)

        for i, col in enumerate(date_part_cols):
            # Clean up the column name for plotting
            clean_col = remove_prefix_suffix(string=col, prefix="ts_", suffix=f"_{time_col}")

            # Get the unique values and their corresponding percentages of occurrences
            counts = df[col].value_counts(normalize=True)[:14]
            values = counts.index
            values_sorted = sorted(values)
            counts_sorted = counts.reindex(values_sorted)
            percentages_sorted = np.round(counts_sorted.values * 100).astype(int)
            percent_matrix = np.array([percentages_sorted])

            # Plot the heatmap of percentages with appropriate title and axis labels
            sns.heatmap(percent_matrix, ax=axs[i * 2], cmap="Blues", annot=True, fmt=".0f", cbar=False)
            axs[i * 2].set_xticklabels(values_sorted)
            axs[i * 2].set_ylabel(clean_col)

            # Get the average of the target variable for each date part
            target_mean = self.target.groupby(df[col]).mean()
            target_mean_sorted = target_mean.reindex(values_sorted)

            # Plot the target variable mean for each date part with appropriate title and axis labels
            axs[i * 2 + 1].plot(target_mean_sorted.index, target_mean_sorted.values, "o-", label='Mean')
            axs[i * 2 + 1].set_xticks(np.arange(len(values_sorted)))
            axs[i * 2 + 1].set_xticklabels(values_sorted)
            axs[i * 2 + 1].set_xlabel(clean_col)
            axs[i * 2 + 1].set_ylabel(self.target_col, color="C1")
            axs[i * 2 + 1].tick_params(axis='y', labelcolor="C1")

        # Set the overall x-axis label and adjust the subplot layout
        fig.suptitle(f"Percentage of Occurrences and Mean {self.target_col} by {time_col} Column Parts:")
        axs[-1].set_xlabel("Values")
        fig.tight_layout()

        # Show the plot
        if plot:
            plt.show()
        plt.close()

    def run(self, plot=True):
        if self.target_type == "numeric":
            print("Numerical statistics for target column")
            NumericStats(self.target_data_info).run(plot=False)
            if plot:
                self.plot_histogram()
                date_cols = self.data_info.feature_types_val["time"]
                if len(date_cols) > 0:
                    time_frequency = self.time_series_handler.time_frequency
                    for date_col in date_cols:
                        print("analyzing date_col", date_col)
                        self.plot_dates(time_col=date_col, time_freq=time_frequency, plot=plot)
                        self.plot_date_parts(time_col=date_col, plot=plot)

        elif self.target_type == "categorical":
            print("Categorical statistics for target column")
            CategoricalStats(self.target_data_info).run(plot=False)
            if plot:
                self.plot_barplot()
