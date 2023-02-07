import math
import random
from math import floor, ceil

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from pandas.errors import SettingWithCopyWarning
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud
from sklearn.manifold import TSNE

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

    def tsne(self):
        if self.n_cols in (1, 2):
            return None
        X = self.data.copy()
        numerical_transformer = SimpleImputer(strategy='median')
        transformer = ColumnTransformer(
            transformers=[
                ('n', numerical_transformer, self.cols)
            ])
        pipeline = Pipeline(steps=[('preprocessor', transformer)])
        X = pipeline.fit_transform(X)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_tsne_3d = TSNE(n_components=3).fit_transform(X)
        X_tsne_2d = X_tsne_3d[:, :2]
        fig = plt.figure(figsize=(16, 8))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, projection='3d')

        if self.data_info.problem_type_val in ["Regression"]:
            ax1.scatter(X_tsne_2d[:, 0], X_tsne_2d[:, 1], c=self.target, cmap='viridis')
            ax2.scatter(X_tsne_3d[:, 0], X_tsne_3d[:, 1], X_tsne_3d[:, 2], c=self.target, cmap='viridis')
            ax1.set_title("Visualizing Numerical Features and Target with t-SNE (2D)")
            ax2.set_title("Visualizing Numerical Features and Target with t-SNE (3D)")
        elif self.data_info.problem_type_val in ["Binary regression", "Binary classification"]:
            target_classes = self.target.unique()
            color_map = {target_class: color for target_class, color in zip(target_classes, ['red', 'blue'])}
            colors = self.target.apply(lambda x: color_map[x])
            ax1.scatter(X_tsne_2d[:, 0], X_tsne_2d[:, 1], c=colors)
            ax2.scatter(X_tsne_3d[:, 0], X_tsne_3d[:, 1], X_tsne_3d[:, 2], c=colors)
            legend_handles = [Patch(color=color_map[target_class], label=target_class) for target_class in
                              target_classes]
            ax1.set_title("Visualizing Numerical Features and Target with t-SNE (2D)")
            ax2.set_title("Visualizing Numerical Features and Target with t-SNE (3D)")
            ax1.legend(handles=legend_handles)
        else:  # Multi-class classification
            target_classes = self.target.unique()
            color_map = {target_class: color for target_class, color in
                         zip(target_classes, plt.cm.rainbow(np.linspace(0, 1, len(target_classes))))}
            colors = self.target.apply(lambda x: color_map[x])
            ax1.scatter(X_tsne_2d[:, 0], X_tsne_2d[:, 1], c=colors)
            ax2.scatter(X_tsne_3d[:, 0], X_tsne_3d[:, 1], X_tsne_3d[:, 2], c=colors)
            legend_handles = [Patch(color=color_map[target_class], label=target_class) for target_class in
                              target_classes]
            ax1.set_title("Visualizing Numerical Features and Target with t-SNE (2D)")
            ax2.set_title("Visualizing Numerical Features and Target with t-SNE (3D)")
            ax1.legend(handles=legend_handles)

    @staticmethod
    def median_imputation(df):
        import warnings
        warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

        median = df.median()
        df.fillna(median, inplace=True)
        return df

    def any_numeric(self):
        return True if len(self.cols) > 0 else False

    def sort_by_pvalue(self):
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
                data_col = self.median_imputation(data_col)
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
                print("Numerical Pvalues plot")
                self.plot_histogram_pvalues(self.cols_sorted_with_pvalue)
        print("Pvalues for top numerical features for chi square test:")
        print(pd.DataFrame(self.cols_sorted_with_pvalue[0:top_features], columns=["feature", "pvalue"]))
        return None

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
        plt.show()
        return None

    @staticmethod
    def plot_wordcloud_pvalues(features_pvalues,
                               title="Numeric Features Pvalues Based on Partial Plot"):
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
        plt.title(title, fontsize=15)
        plt.show()
        return None

    def partial_plot(self, classification_row_percent=True):
        import warnings
        warnings.simplefilter("ignore")
        top_features = 25
        if self.n_cols <= 25:
            sample_features = self.n_cols
            top_features = self.n_cols
        else:
            sample_features = min(2 * 25, int(self.n_cols / 2))
        top_feature_names = random.sample(self.cols_sorted[0:sample_features], top_features)
        feature_index = {feature: index for index, feature in enumerate(self.cols_sorted)}
        top_feature_names.sort(key=lambda x: feature_index[x])

        num_plots = len(top_feature_names)
        dim = int(math.sqrt(num_plots))
        num_rows = math.ceil(num_plots / dim)
        fig, ax = plt.subplots(num_rows, dim)
        if self.data_info.problem_type_val in ["Binary regression"]:
            plt.suptitle("Partial Plot to Identify Patterns between Sampled Numeric Features and Target",
                         fontsize=16, fontweight='bold')
            for i in range(len(top_feature_names)):
                col = top_feature_names[i]
                column = self.data[col]
                target = self.target
                ax_i = ax[i // dim, i % dim]
                sns.regplot(x=column, y=target, logistic=True, n_boot=250, y_jitter=.03, ax=ax_i)
                ax_i.set_xlabel("")
                ax_i.set_ylabel("")
                ax_i.set_title(col, fontweight='bold', transform=ax_i.transAxes, y=0.5)
            plt.show()
        if self.data_info.problem_type_val in ["Regression"]:
            plt.suptitle("Partial Plot to Identify Patterns between Sampled Numeric Features and Target", fontsize=16,
                         fontweight='bold')
            grid_size = 4
            num_features = min(grid_size * grid_size, top_features)
            num_rows = int(np.ceil(num_features / grid_size))
            for i, col in enumerate(top_feature_names[:num_features]):
                plt.subplot(num_rows, grid_size, i + 1)
                column = self.data[col]
                target = self.target
                plt.scatter(column, target)
                plt.xlabel(col)
                plt.ylabel(self.data_info.target)
            plt.show()
        elif self.data_info.problem_type_val in ["Multiclass classification", "Binary classification"]:
            if classification_row_percent:
                plt.suptitle("Partial Plot to Identify Patterns between Sampled Numeric Features and Target\n"
                             "Showing % from cluster (row)",
                             fontsize=14, fontweight='bold')
            else:
                plt.suptitle("Partial Plot to Identify Patterns between Sampled Numeric Features and Target\n"
                             "Showing % from Target (column)",
                             fontsize=14, fontweight='bold')
            for i in range(len(top_feature_names)):
                if i < 9:
                    plt.subplot(3, 3, i + 1)
                    col = top_feature_names[i]
                    data_col = self.data[[col]]
                    num_groups = min(floor(self.data_info.rows / 20), 10)
                    kmeans = KMeans(n_clusters=num_groups, n_init=10)
                    if self.data[col].isna().any():
                        data_col = self.median_imputation(data_col)
                    kmeans.fit(data_col)
                    labels = kmeans.labels_
                    top_5_target_values = self.target.value_counts().index[:5]
                    target_filtered = self.target[self.target.isin(top_5_target_values)]
                    labels_filtered = labels[self.target.isin(top_5_target_values)]
                    contingency_table = pd.crosstab(index=labels_filtered, columns=target_filtered)
                    if classification_row_percent:
                        contingency_table_pct = contingency_table.div(contingency_table.sum(1), axis=0)
                    else:
                        contingency_table_pct = contingency_table.div(contingency_table.sum(0), axis=1)
                    sns.heatmap(contingency_table_pct, annot=False, cmap='Blues')
                    plt.ylabel(col, fontsize=8, fontweight='bold')
                    plt.xlabel(None)
            plt.show()
            return None

    def run(self):
        if self.n_cols > 1:
            self.analyze_pvalue()
            if 'classification' in self.data_info.problem_type_val.lower():
                self.partial_plot(classification_row_percent=True)
                # self.partial_plot(classification_row_percent=False)
            else:
                self.partial_plot()
            self.tsne()
        elif self.n_cols == 1:
            self.analyze_pvalue()
            if 'classification' in self.data_info.problem_type_val.lower():
                self.partial_plot(classification_row_percent=True)
                self.partial_plot(classification_row_percent=False)
            else:
                self.partial_plot()
        return None


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
