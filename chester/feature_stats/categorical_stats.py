import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from chester.zero_break.problem_specification import DataInfo


class CategoricalStats:
    def __init__(self, data_info: DataInfo):
        self.data_info = data_info
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

    def plot_value_counts(self, n=25):
        if not self.any_categorical():
            return None
        top_n = self.cols_sorted[:min(len(self.cols_sorted), n)]
        if len(top_n) == 1:
            col = top_n[0]
            fig, ax = plt.subplots(1, 1, figsize=(20, 5))
            cat_col = self.data[col].apply(lambda x: "cat " + str(x))
            data = cat_col.value_counts()
            sns.barplot(x=data.index[:5], y=data.values[:5], ax=ax)
            plot_title = f"{col}"
            ax.set_title(plot_title)
            ax.set_xlabel(None)
            # return None
        else:
            fig, ax = plt.subplots(1, len(top_n), figsize=(20, 5))
            for i, col in enumerate(top_n):
                data = self.data[col].value_counts()
                # print("wow wow!!", data.index[:5])
                # print("wow wow!!", data.values[:5])
                plot_title = f"{col}"
                sns.barplot(x=data.index[:5], y=data.values[:5], ax=ax[i])
                ax[i].set_title(plot_title)
                ax[i].set_xlabel(None)
            plt.suptitle("Top 5 Value Counts for Each Feature")
            plt.show()

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

            # 1. in Distribution you need to print the value like this: A:countA, B:countB.
            dist_str = ', '.join([f"{row['index']}: {row['percentage']:.0f}%" for _, row in value_counts.iterrows()])
            result_dicts.append(
                {'col': col, '# unique': unique_values, '# missing': missing_values, 'Distribution': dist_str})

            # 2. In addition sample 3 values out of the distinct values of the column and add it to the result df
            data_unique_values = data.dropna().drop_duplicates()
            col_len = len(data_unique_values)
            sample_values = data_unique_values.sample(min(col_len, 3)).values
            result_dicts[-1]['Sample'] = ', '.join(sample_values)

            # add more columns
            # 1. % from all that covers the top 5 values
            top_5 = 100 * value_counts.iloc[:5]["count"].sum() / value_counts["count"].sum()
            result_dicts[-1][f'Top 5 values coverage'] = f"{top_5:.0f}%"

        results_df = pd.DataFrame(result_dicts)

        if is_print:
            print(format_df(results_df))
        return results_df

    def run(self):
        self.calculate_stats()
        self.plot_value_counts()


def format_df(df, max_value_width=12, distribution_max_value_width=20, distribution_col="Distribution"):
    pd.options.display.max_columns = None

    def trim_value(val):
        if len(str(val)) > max_value_width:
            return str(val)[:max_value_width] + "..."
        return str(val)

    def trim_distribution_value(val):
        if len(str(val)) > distribution_max_value_width:
            return str(val)[:distribution_max_value_width] + "..."
        return str(val)

    df_subset = df.drop(distribution_col, axis=1)
    df_subset = df_subset.applymap(trim_value)
    df[df_subset.columns] = df_subset
    df[distribution_col] = df[distribution_col].apply(trim_distribution_value)

    return df
