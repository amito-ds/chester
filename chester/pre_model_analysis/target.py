import matplotlib.pyplot as plt
import numpy as np
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
        plt.hist(target, bins=30, edgecolor='k')
        plt.figure(figsize=(10, 10))
        plt.rcParams.update({'font.size': 12})
        plt.xlabel('Values')
        plt.ylabel('Counts')
        plt.title(f'Histogram of {self.target.name}')
        plt.show()

    def plot_barplot(self):
        target = self.target
        value_counts = target.value_counts()
        percentages = np.array(value_counts / target.size * 100)
        fig, ax1 = plt.subplots()
        plt.figure(figsize=(10, 10))
        plt.rcParams.update({'font.size': 12})
        ax2 = ax1.twinx()
        ax1.bar(value_counts.index, value_counts.values, color='gray')
        ax1.set_ylabel('Counts', color='gray')
        ax2.plot(value_counts.index, percentages, color='red', marker='o')
        ax2.set_ylabel('Percentages', color='red')
        ax1.set_xlabel('Values')
        plt.title(f'Bar Plot of {self.target.name}')
        plt.show()

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