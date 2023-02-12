import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas.errors import SettingWithCopyWarning

from chester.feature_stats.categorical_stats import CategoricalStats
from chester.feature_stats.numeric_stats import NumericStats
from chester.zero_break.problem_specification import DataInfo


class TargetPreModelAnalysis:
    def __init__(self, data_info: DataInfo):
        import warnings
        warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
        self.data_info = data_info
        self.target = self.data_info.data[self.data_info.target]
        self.target_col = self.data_info.target
        self.target_df = self.data_info.data[[self.data_info.target]]
        self.target_df.rename(columns={self.data_info.target: 'target_label'}, inplace=True)
        self.data_info = DataInfo(data=self.target_df)
        self.data_info.calculate()
        self.target_type = None
        # target type
        for feature_type, features in self.data_info.feature_types_val.items():
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
        target = self.target
        value_counts = target.value_counts()
        percentages = np.array(value_counts / target.size * 100)
        fig, ax1 = plt.subplots(figsize=(7, 7), dpi=100)
        ax2 = ax1.twinx()

        # Create a color map that shows the percentage of each target value
        cmap = sns.light_palette("green", as_cmap=True)
        heatmap = np.array([percentages, ] * len(value_counts))

        # Create a bar plot with a heatmap color scheme
        ax1.barh(value_counts.index, value_counts.values, color=cmap(heatmap))
        ax1.set_xlabel('Counts')
        ax1.set_ylabel('Values')
        ax1.invert_yaxis()

        # Set the y-axis tick labels to the target values
        ax1.set_yticklabels(value_counts.index)

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

    def run(self, plot=True):
        if self.target_type == "numeric":
            print("Numerical statistics for target column")
            NumericStats(self.data_info).run(plot=False)
            if plot:
                self.plot_histogram()
        elif self.target_type == "categorical":
            print("Categorical statistics for target column")
            CategoricalStats(self.data_info).run(plot=False)
            if plot:
                self.plot_barplot()
