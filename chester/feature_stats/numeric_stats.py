import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from chester.zero_break.problem_specification import DataInfo


class NumericStats:
    def __init__(self, data_info: DataInfo, max_print=None):
        self.data_info = data_info
        self.max_print = max_print
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

    def plot_correlation(self, n=25, plot=True):
        if not self.any_numeric():
            return None
        if not plot:
            return None
        top_n = self.cols_sorted[:min(len(self.cols_sorted), 3 * n)]
        top_n_sampled = top_n[:min(len(top_n), n)]
        data = self.data.sort_values(by=top_n_sampled)
        corr = data[top_n_sampled].corr()
        if len(self.cols_sorted) <= n:
            plot_title = f"Pearson Correlation Plot"
        else:
            plot_title = f"Pearson Correlation Plot for {n} randomly sampled Features"
        plt.figure(figsize=(13, 13))
        plt.rcParams.update({'font.size': 18})
        sns.heatmap(corr, annot=False)
        plt.title(plot_title)
        print("Matrix correlation for numerical features")
        print("""\n
        💡 Rule of thumb:
        👍 Strong positive correlation: >= 0.7
        🤔 Moderate positive correlation: between 0.5 and 0.7
        🤨 Weak positive correlation: between 0.3 and 0.5
        🤷‍️ No/Negligible correlation: < 0.3
        """)
        plt.show()
        plt.close()

    def calculate_stats(self, is_print=True):
        from chester.util import ReportCollector, REPORT_PATH
        rc = ReportCollector(REPORT_PATH)

        if self.data.select_dtypes(include=[np.number]).empty:
            return None

        result_dicts = []
        for col in self.cols:
            col_data = self.data[col]
            stats = compute_statistics(col_data)
            result_dicts.append({
                'col': col,
                '# unique': stats['unique'],
                '# missing': stats['missing'],
                'max': stats['max'],
                'min': stats['min'],
                'avg': stats['avg'],
                'std': stats['std'],
                'CI': stats['CI'],
                'median': stats['median'],
                'top_vals': stats['top_vals'],
                'bottom_vals': stats['bottom_vals']
            })

        results_df = pd.DataFrame(result_dicts)

        if is_print:
            formatted_df = results_df
            if self.max_print is not None:
                # Assuming the function 'format_df' exists and is used to format the dataframe
                formatted_df = format_df(df=results_df, max_value_width=self.max_print)
            print(formatted_df)
            len_df = len(formatted_df)
            rc.save_object(obj=formatted_df.sample(min(len_df, 10)), text="Feature stats:")

        return results_df

    def run(self, plot=True):
        self.calculate_stats()
        self.plot_correlation(plot=plot)
        return None


def format_df(df, max_value_width=25,
              col_max_value_width=25,
              ci_max_value_width=25,
              ci_col="CI", col_col="col"):
    pd.options.display.max_columns = None

    def trim_value(val):
        if len(str(val)) > max_value_width:
            return str(val)[:max_value_width] + "..."
        return str(val)

    def trim_ci_value(val):
        if len(str(val)) > ci_max_value_width:
            return str(val)[:ci_max_value_width] + "..."
        return str(val)

    def trim_col_value(val):
        if len(str(val)) > col_max_value_width:
            return str(val)[:ci_max_value_width] + "..."
        return str(val)

    df_subset = df.drop([ci_col, col_col], axis=1)
    df_subset = df_subset.applymap(trim_value)
    df[df_subset.columns] = df_subset
    df[ci_col] = df[ci_col].apply(trim_ci_value)
    df[col_col] = df[col_col].apply(trim_col_value)

    return df


def round_value(value):
    """Custom rounding function to handle integer and non-integer values."""
    if isinstance(value, (int, np.integer)):
        return value
    else:
        return round(value, 2)


def compute_statistics(data):
    """Compute statistics for a given data series."""
    data_drop_dups = data.drop_duplicates()
    unique_values = data.nunique()
    missing_values = data.isnull().sum()
    n = len(data)

    # If all values are missing, return None statistics
    if missing_values == n:
        return {
            'unique': None,
            'missing': missing_values,
            'max': None,
            'min': None,
            'avg': None,
            'std': None,
            'CI': (None, None),
            'median': None,
            'top_vals': None,
            'bottom_vals': None
        }

    max_vals = round_value(data_drop_dups.max())
    min_vals = round_value(data_drop_dups.min())
    avg_vals = round_value(data.mean())
    std_vals = round_value(data.std())
    ci_vals = (round_value(avg_vals - 1.645 * (std_vals / np.sqrt(n))),
               round_value(avg_vals + 1.645 * (std_vals / np.sqrt(n))))
    median_vals = round_value(data.median())
    top_vals = ",".join(map(str, data_drop_dups.nlargest(3).apply(round_value).tolist()))
    bottom_vals = ",".join(map(str, data_drop_dups.nsmallest(3).apply(round_value).tolist()))

    return {
        'unique': unique_values,
        'missing': missing_values,
        'max': max_vals,
        'min': min_vals,
        'avg': avg_vals,
        'std': std_vals,
        'CI': ci_vals,
        'median': median_vals,
        'top_vals': top_vals,
        'bottom_vals': bottom_vals
    }
