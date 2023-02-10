import math

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from chester.zero_break.problem_specification import DataInfo


class CategoricalStats:
    def __init__(self, data_info: DataInfo, max_print=None):
        self.data_info = data_info
        self.max_print = max_print
        self.cols = self.data_info.feature_types_val["categorical"]
        self.data = self.data_info.data[self.cols]
        self.cols_sorted = self.sort_by_cardinality()

    def any_categorical(self):
        return True if len(self.cols) > 0 else False

    def sort_by_cardinality(self):
        if not self.any_categorical():
            return []
        cardinalities = []
        for col in self.cols:
            data = self.data[col]
            cardinalities.append((col, data.nunique()))
        sorted_cardinalities = sorted(cardinalities, key=lambda x: x[1], reverse=True)
        return [x[0] for x in sorted_cardinalities]

    def sample_top_features(self, n):
        top_features = self.cols_sorted[:3 * n]
        return top_features

    def plot_value_counts(self, n=25, norm=True, plot=True):
        if not self.any_categorical():
            return None
        if not plot:
            return None
        top_n = self.cols_sorted[:min(len(self.cols_sorted), n)]
        num_plots = len(top_n)
        if num_plots == 1:
            col = top_n[0]
            fig, ax = plt.subplots(1, 1, figsize=(20, 5))
            cat_col = self.data[col].apply(lambda x: "cat " + str(x))
            data = cat_col.value_counts(normalize=norm)
            sns.barplot(x=data.index[:5], y=data.values[:5], ax=ax)
            plot_title = f"{col}"
            ax.set_title(plot_title)
            ax.set_xlabel(None)
            ax.set_ylim(0, 1)
            return None
        else:
            dim = math.ceil(math.sqrt(len(top_n)))
            num_rows = math.ceil(num_plots / dim)
            fig, ax = plt.subplots(num_rows, dim)
            fig.tight_layout()
            if norm:
                fig.suptitle("Top 5 Value % for Each Feature")
            else:
                fig.suptitle("Top 5 Value Counts for Each Feature")
            for i, col in enumerate(top_n):
                data = pd.DataFrame(self.data[col].value_counts(normalize=norm)[0:5]).reset_index(drop=False)
                plot_title = f"{col}"
                ax_i = ax[i // dim, i % dim]
                sns.barplot(x=data.iloc[:, 0], y=data.iloc[:, 1].to_list(), ax=ax_i)
                ax_i.set_title(plot_title)
                ax_i.set_xlabel(None)
                if norm:
                    ax_i.set_ylim(0, 1)
            plt.tight_layout()
            plt.show()
            return None

    def calculate_stats(self, is_print=True):
        if not self.any_categorical():
            return None
        result_dicts = []
        for col in self.cols:
            data = self.data[col]
            unique_values = data.nunique()
            missing_values = data.isnull().sum()
            value_counts = data.value_counts().rename("count").reset_index()
            value_counts["percentage"] = 100 * value_counts["count"] / value_counts["count"].sum()
            value_counts = value_counts.sort_values("count", ascending=False)
            dist_str = ', '.join([f"{row['index']}: {row['percentage']:.0f}%" for _, row in value_counts.iterrows()])
            result_dicts.append(
                {'col': col, '# unique': unique_values, '# missing': missing_values, 'Distribution': dist_str})
            data_unique_values = data.dropna().drop_duplicates()
            col_len = len(data_unique_values)
            values_to_sample = 3
            if col_len < values_to_sample:
                values_to_sample = col_len
            sample_values = [str(value) for value in data_unique_values.sample(min(col_len, values_to_sample)).values]
            result_dicts[-1]['Sample'] = ', '.join(sample_values)

            # add more columns
            # 1. % from all that covers the top 5 values
            top_5 = 100 * value_counts.iloc[:5]["count"].sum() / value_counts["count"].sum()
            result_dicts[-1][f'Top 5 values coverage'] = f"{top_5:.0f}%"

        results_df = pd.DataFrame(result_dicts)

        if is_print:
            if self.max_print is not None:
                print(format_df(df=results_df,
                                max_value_width=self.max_print,
                                ))
            else:
                print(format_df(results_df))
        return results_df

    def run(self, plot=True):
        self.calculate_stats()
        self.plot_value_counts(norm=True, plot=plot)
        self.plot_value_counts(norm=False, plot=plot)
        return None


def format_df(df, max_value_width=30):
    pd.options.display.max_columns = None

    def trim_value(val):
        if len(str(val)) > max_value_width:
            return str(val)[:max_value_width] + "..."
        return str(val)

    df = df.applymap(trim_value)

    return df
