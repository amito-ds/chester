from math import floor

import matplotlib.pyplot as plt
import pandas as pd
from pandas.errors import SettingWithCopyWarning
from wordcloud import WordCloud

from chester.zero_break.problem_specification import DataInfo


class NumericPreModelAnalysis:
    def __init__(self, data_info: DataInfo):
        self.data_info = data_info
        self.cols = self.data_info.feature_types_val["numeric"]
        self.n_cols = len(self.cols)
        self.target = self.data_info.data[self.data_info.target]
        self.target_df = self.data_info.data[[self.data_info.target]]
        self.data = self.data_info.data[self.cols]
        # calc
        self.cols_sorted = self.sort_by_pvalue()
        self.cols_sorted_with_pvalue = None

    @staticmethod
    def median_imputation(df, col):
        import warnings
        warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

        median = df[col].median()
        df[col].fillna(median, inplace=True)
        return df

    def any_numeric(self):
        return True if len(self.cols) > 0 else False

    def sort_by_pvalue(self):
        from sklearn.cluster import KMeans
        from scipy.stats import chi2_contingency
        import warnings
        warnings.simplefilter("ignore")

        problem_type = self.data_info.problem_type_val

        if problem_type in ["Regression"]:
            num_groups = min(floor(self.data_info.rows / 20), 10)
            kmeans = KMeans(n_clusters=num_groups, n_init=10)
            kmeans.fit(self.target_df)
            target_labels = kmeans.labels_

        feature_pvalue_list = []
        for col in self.cols:
            data_col = self.data[[col]]
            num_groups = min(floor(self.data_info.rows / 20), 10)
            kmeans = KMeans(n_clusters=num_groups, n_init=10)
            if self.data[col].isna().any():
                data_col = self.median_imputation(self.data, col)
            kmeans.fit(data_col)
            labels = kmeans.labels_
            if problem_type == "Regression":
                contingency_table = pd.crosstab(index=labels, columns=target_labels)
                chi2, pvalue, _, _ = chi2_contingency(contingency_table)
            else:
                contingency_table = pd.crosstab(index=labels, columns=self.target)
                chi2, pvalue, _, _ = chi2_contingency(contingency_table)
            feature_pvalue_list.append((col, pvalue))

        sorted_list = sorted(feature_pvalue_list, key=lambda x: x[1], reverse=False)
        self.cols_sorted_with_pvalue = sorted_list
        return [x[0] for x in sorted_list]

    def analyze_pvalue(self, is_plot=True, top_features=10):
        self.sort_by_pvalue()
        self.plot_wordcloud_pvalues(self.cols_sorted_with_pvalue)
        if is_plot:
            if self.n_cols > 50:
                print("plotting!")
                self.plot_histogram_pvalues(self.cols_sorted_with_pvalue)
        print("Pvalues for top features:")
        print(pd.DataFrame(self.cols_sorted_with_pvalue[0:top_features], columns=["feature", "pvalue"]))

    @staticmethod
    def plot_histogram_pvalues(features_pvalues):
        """
        Plot histogram of p-values for features.
        :param features_pvalues: List of tuples (column name, pvalue).
        :return: None.
        """
        pvalues = [pvalue for _, pvalue in features_pvalues]
        fig, ax = plt.subplots()
        ax.hist(pvalues, bins=50, edgecolor='k', color='#2ecc71')
        ax.set_title("Histogram of P-values for Numerical Features", fontsize=16)
        ax.set_xlabel("P-value", fontsize=14)
        ax.set_ylabel("Frequency", fontsize=14)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.tick_params(axis='both', which='both', labelsize=12)
        plt.show(block=False)

    @staticmethod
    def plot_wordcloud_pvalues(features_pvalues,
                               title="Features Pvalues Based on Partial Plot"):
        """
        Plot word cloud of features weighted by their p-value.
        :param features_pvalues: List of tuples (column name, pvalue).
        :param title: Title of the plot.
        :return: None.
        """
        features_pvalues = [(feature, 1 - pvalue) for feature, pvalue in features_pvalues]
        wordcloud = WordCloud(
            random_state=21,
            normalize_plurals=True).generate_from_frequencies(dict(features_pvalues))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.title(title, fontsize=15)
        plt.show(block=False)


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
