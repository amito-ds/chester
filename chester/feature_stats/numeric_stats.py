import numpy as np

from chester.feature_stats.utils import round_value
from chester.zero_break.problem_specification import DataInfo
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


class NumericStats:
    def __init__(self, data_info: DataInfo):
        self.data_info = data_info
        self.cols = self.data_info.feature_types_val["numeric"]
        self.data = self.data_info.data[self.cols]
        self.cols_sorted = self.sort_by_variance()

    def any_numeric(self):
        return True if len(self.cols) > 0 else False

    def sort_by_variance(self):
        if not self.any_numeric():
            return []
        variances = []
        for col in self.cols:
            data = self.data[col]
            variances.append((col, data.var()))
        sorted_variances = sorted(variances, key=lambda x: x[1], reverse=True)
        return [x[0] for x in sorted_variances]

    def plot_correlation(self, n=25):
        if not self.any_numeric():
            return None
        top_n = self.cols_sorted[:min(len(self.cols_sorted), 3 * n)]
        top_n_sampled = top_n[:min(len(top_n), n)]
        data = self.data.sort_values(by=top_n_sampled)
        corr = data[top_n_sampled].corr()
        if len(self.cols_sorted) <= n:
            plot_title = f"Pearson Correlation Plot"
        else:
            plot_title = f"Pearson Correlation Plot for {n} randomly sampled Features"
        sns.heatmap(corr, annot=False)
        plt.title(plot_title)
        print("Plotting matrix correlation")
        print("""\n
        Rule of thumbs:
        1. Strong positive correlation: >= 0.7
        2. Moderate positive correlation: between 0.5 and 0.7
        3. Weak positive correlation: between 0.3 and 0.5
        4. No/Negligible correlation: < 0.3
        """)
        plt.show()

    def calculate_stats(self, is_print=True):
        if not self.any_numeric():
            return None
        result_dicts = []
        for col in self.cols:
            data = self.data[col]
            data_drop_dups = data.drop_duplicates()
            unique_values = data.nunique()
            missing_values = data.isnull().sum()
            max_vals = data_drop_dups.max()
            min_vals = data_drop_dups.min()
            avg_vals = data.mean()
            std_vals = data.std()
            n = len(data)
            ci_vals = [avg_vals - 1.645 * (std_vals / np.sqrt(n)), avg_vals + 1.645 * (std_vals / np.sqrt(n))]
            median_vals = data.median()
            top_vals = data_drop_dups.nlargest(3).apply(round_value)
            bottom_vals = data_drop_dups.nsmallest(3).apply(round_value)
            max_vals = max_vals.round(2)
            min_vals = min_vals.round(2)
            avg_vals = avg_vals.round(2)
            std_vals = std_vals.round(2)
            ci_vals = round_value(ci_vals[0]), round_value(ci_vals[1])
            median_vals = median_vals.round(2)
            top_vals = ",".join(map(str, top_vals.tolist()))
            bottom_vals = ",".join(map(str, bottom_vals.tolist()))
            result_dicts.append({'col': col, '# unique': unique_values, '# missing': missing_values,
                                 'max': max_vals,
                                 'min': min_vals, 'avg': avg_vals, 'std': std_vals, 'CI': ci_vals,
                                 'median': median_vals,
                                 'top_vals': top_vals, 'bottom_vals': bottom_vals})
        results_df = pd.DataFrame(result_dicts)
        if is_print:
            print(format_df(results_df))
        return results_df

    def run(self):
        self.calculate_stats()
        self.plot_correlation()

    # plots:
    # matrix corr pearson
    # for top 9 features with the highest var: plot 3X3 histogram


def format_df(df, max_value_width=10, ci_max_value_width=15, ci_col="CI"):
    pd.options.display.max_columns = None

    def trim_value(val):
        if len(str(val)) > max_value_width:
            return str(val)[:max_value_width] + "..."
        return str(val)

    def trim_ci_value(val):
        if len(str(val)) > ci_max_value_width:
            return str(val)[:ci_max_value_width] + "..."
        return str(val)

    df_subset = df.drop(ci_col, axis=1)
    df_subset = df_subset.applymap(trim_value)
    df[df_subset.columns] = df_subset
    df[ci_col] = df[ci_col].apply(trim_ci_value)

    return df
