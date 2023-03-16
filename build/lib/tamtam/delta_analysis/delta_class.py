import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

from tamtam.ab_info.ab_class import ABInfo
from tamtam.user_class.user_class import TestInfo
from tamtam.utils.column_utils import Columns


class DeltaPValueCalc:
    def __init__(self, ab_info: ABInfo, test_info: TestInfo):
        self.ab_info = ab_info
        self.test_info = test_info
        self.df = self.ab_info.df
        self.side_col = self.ab_info.test_info.get_side_col()[0]
        self.control = "A"
        self.treatment = "B"
        self.trimmed = self.ab_info.trimmed
        self.trimmed_cols = self.ab_info.trimmed_cols
        self.trimmed_col_by_metric = self.ab_info.trimmed_col_by_metric

    @staticmethod
    def weighted_ttest_1samp(data, popmean, weights):
        # Calculate the weighted mean and standard deviation of the sample
        mean = np.average(data, weights=weights)
        std = np.sqrt(np.average((data - mean) ** 2, weights=weights))

        # Calculate the weighted t-test statistic and p-value
        t_stat = (mean - popmean) / (std / np.sqrt(len(data)))
        p_val = stats.t.sf(np.abs(t_stat), len(data) - 1) * 2

        return t_stat, p_val

    def run_single(self, metric_col):
        weight_col = self.test_info.get_weight_col() or Columns.weight
        is_higher_better = self.test_info.is_higher_better  # saying it should be a test >0 or <=0

        # Subset the data and convert to numpy arrays
        data = self.trimmed[[weight_col, metric_col]].to_numpy()

        # Calculate the weighted mean and standard deviation for the single group
        mean = np.average(data[:, 1], weights=data[:, 0])
        std = np.sqrt(np.average((data[:, 1] - mean) ** 2, weights=data[:, 0]))

        # Calculate the t-test statistic and p-value for the null hypothesis that the mean is 0
        if is_higher_better:
            t_stat, p_val = self.weighted_ttest_1samp(data[:, 1], 0, data[:, 0])
        else:
            t_stat, p_val = self.weighted_ttest_1samp(data[:, 1], 0, data[:, 0])
            t_stat = -t_stat

        # Calculate the effect size, MDE, and CI
        if is_higher_better:
            effect_size = mean
            mde = stats.t.ppf(0.8, len(data) - 1) * std / np.sqrt(len(data))
            ci = stats.t.interval(0.95, len(data) - 1, loc=mean, scale=std / np.sqrt(len(data)))
        else:
            effect_size = -mean
            mde = stats.t.ppf(0.8, len(data) - 1) * std / np.sqrt(len(data))
            ci = stats.t.interval(0.95, len(data) - 1, loc=-mean, scale=std / np.sqrt(len(data)))

        results = {
            'metric': metric_col,
            'weighted_mean': mean,
            'weighted_std': std,
            't_statistic': t_stat,
            'p_value': p_val,
            'effect_size': effect_size,
            'mde': mde,
            'ci_lower': ci[0],
            'ci_upper': ci[1]
        }
        return results

    def run_all_metrics(self):
        single_results = []
        for col in self.trimmed_cols:
            single_results.append(self.run_single(col))
        return single_results

    def plot_single_metric(self, col, delta_results):
        trimmed_cols = self.trimmed_col_by_metric[col]
        test_results = delta_results[delta_results.metric.isin(trimmed_cols)]
        self._plot_util(test_results)

    def plot(self, delta_results):
        for col in self.ab_info.get_metric_cols():
            self.plot_single_metric(col, delta_results)

    @staticmethod
    def _plot_util(df):
        # Extract the unique x-axis values from the DataFrame (corresponding to different trim levels)
        x_values = df['metric'].str.rsplit('_', n=1, expand=True)[1].astype(float)

        # Create a figure with 3 rows and 2 columns of subplots
        fig, axs = plt.subplots(3, 2, figsize=(14, 9))
        plt.title("Significant Calculation")

        # Plot the mean and confidence interval in the top left subplot
        axs[0, 0].errorbar(x_values, df['weighted_mean'],
                           yerr=[df['weighted_mean'] - df['ci_lower'], df['ci_upper'] - df['weighted_mean']],
                           label='Metric diff', marker='o')
        axs[0, 0].plot(x_values, df['weighted_mean'], linestyle='-', label='Weighted mean', color='blue')
        axs[0, 0].plot(x_values, df['ci_upper'], linestyle='--', label='CI Upper', color='red')
        axs[0, 0].plot(x_values, df['ci_lower'], linestyle='--', label='CI Lower', color='green')
        axs[0, 0].set_xlabel('Trim level')
        axs[0, 0].set_ylabel(df['metric'].iloc[0])
        axs[0, 0].legend()

        # Plot the p-value in the top right subplot
        axs[0, 1].plot(x_values, df['p_value'], label='p-value', marker='o')
        axs[0, 1].set_xlabel('Trim level')
        axs[0, 1].set_ylabel('p-value')
        axs[0, 1].legend()

        # Plot the MDE in the bottom left subplot
        axs[1, 0].plot(x_values, df['mde'], label='MDE', marker='o')
        axs[1, 0].set_xlabel('Trim level')
        axs[1, 0].set_ylabel('MDE')
        axs[1, 0].legend()

        # Plot the effect size in the bottom right subplot
        axs[1, 1].plot(x_values, df['effect_size'], label='Effect size', marker='o')
        axs[1, 1].set_xlabel('Trim level')
        axs[1, 1].set_ylabel('Effect size')
        axs[1, 1].legend()

        # Plot the standard deviation in the bottom middle subplot
        axs[2, 0].plot(x_values, df['weighted_std'], label='Std(delta(metric))', marker='o')
        axs[2, 0].set_xlabel('Trim level')
        axs[2, 0].set_ylabel('Std(delta(metric))')
        axs[2, 0].legend()

        # Remove any unused subplots
        axs[2, 1].remove()

        # Adjust the spacing between subplots and display the plot
        fig.tight_layout()
        plt.show()
        plt.close()

    def run(self):
        print("Delta, PValue, effect size:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)

        single_results = self.run_all_metrics()  # get diff, pvalue
        print("Test Statistics:")
        print(pd.DataFrame(single_results))

        # plot
        self.plot(delta_results=pd.DataFrame(single_results))
