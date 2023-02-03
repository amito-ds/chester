import pandas as pd
import numpy as np

from chester.zero_break.problem_specification import DataInfo


def round_value(x):
    if abs(x) < 10:
        return round(x, 2)
    elif abs(x) < 100:
        return round(x, 1)
    else:
        return round(x)


class NumericStats:
    def __init__(self, data_info: DataInfo):
        self.data_info = data_info
        self.cols = self.data_info.feature_types_val["numeric"]
        self.data = self.data_info.data[self.cols]

    def calculate_stats(self, is_print=True):
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
            format_df(results_df)
        return results_df

    # to calculate per feature:
    # number of unique values
    # of missing
    # max, min, std, CI (90%)
    # avg, median
    # outliers detection: top/bottom 3 values
    # all numbers rounded to 2 digits
    # plots:
    # matrix corr pearson
    # for top 9 features with the highest var: plot 3X3 histogram


import pandas as pd


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

    print(df)
    return df
    # return df.to_string(index=False, columns=False)
