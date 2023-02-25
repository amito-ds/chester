import math
import random
from math import floor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.manifold import TSNE
from wordcloud import WordCloud

from chester.zero_break.problem_specification import DataInfo


class CategoricalPreModelAnalysis:
    def __init__(self, data_info: DataInfo):
        self.data_info = data_info
        self.cols = list(set(self.data_info.feature_types_val["categorical"]))
        # drop dups
        if self.data_info.data.columns.duplicated().any():
            drop_dup_data = self.data_info.data.loc[:, ~ self.data_info.data.columns.duplicated()]
            self.data = drop_dup_data
            self.data_info.data = drop_dup_data
        self.n_cols = len(self.cols)
        self.target = self.data_info.data[self.data_info.target]
        self.target_df = self.data_info.data[[self.data_info.target]]
        self.data = self.data_info.data[self.cols]
        # calc
        self.cols_sorted = self.sort_by_pvalue()
        self.cols_sorted_with_pvalue = None

    def tsne(self):
        X = self.data.copy()
        X = X.sample(n=min(5000, len(X)))
        target = self.target[X.index]

        X_tsne_3d = TSNE(n_components=3).fit_transform(pd.get_dummies(X))
        X_tsne_2d = X_tsne_3d[:, :2]

        fig = plt.figure(figsize=(16, 10))
        ax1 = plt
        # ax2 = fig.add_subplot(122, projection='3d')
        if self.data_info.problem_type_val in ["Regression"]:
            ax1.hexbin(X_tsne_2d[:, 0], X_tsne_2d[:, 1], C=target, gridsize=50, cmap='viridis', edgecolors='black')
            # ax2.scatter(X_tsne_3d[:, 0], X_tsne_3d[:, 1], X_tsne_3d[:, 2], c=self.target, cmap='viridis')
            ax1.title("Visualizing Categorical Features and Target with t-SNE (2D)")
            # ax2.set_title("Visualizing Categorical Features and Target with t-SNE (3D)")
        elif self.data_info.problem_type_val in ["Binary regression", "Binary classification"]:
            target_classes = target.unique()
            color_map = {target_class: color for target_class, color in zip(target_classes, ['red', 'blue'])}
            colors = target.apply(lambda x: color_map[x])
            ax1.scatter(X_tsne_2d[:, 0], X_tsne_2d[:, 1], c=colors)
            # ax2.scatter(X_tsne_3d[:, 0], X_tsne_3d[:, 1], X_tsne_3d[:, 2], c=colors)
            legend_handles = [Patch(color=color_map[target_class], label=target_class) for target_class in
                              target_classes]
            ax1.title("Visualizing Categorical Features and Target with t-SNE (2D)")
            # ax2.set_title("Visualizing Categorical Features and Target with t-SNE (3D)")
            ax1.legend(handles=legend_handles)
        else:  # Multi-class classification
            target_classes = target.unique()
            color_map = {target_class: color for target_class, color in
                         zip(target_classes, plt.cm.rainbow(np.linspace(0, 1, len(target_classes))))}
            ax1.legend(
                handles=[Patch(color=color_map[target_class], label=target_class) for target_class in target_classes])
            # ax2.legend(
            #     handles=[Patch(color=color_map[target_class], label=target_class) for target_class in target_classes])
        plt.show()
        plt.close()

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
            data_col = self.data[col]
            if problem_type == "Regression":
                contingency_table = pd.crosstab(data_col, columns=target_labels)
                chi2, pvalue, _, _ = chi2_contingency(contingency_table)
            else:
                contingency_table = pd.crosstab(index=data_col, columns=self.target)
                chi2, pvalue, _, _ = chi2_contingency(contingency_table)
            feature_pvalue_list.append((col, pvalue))

        sorted_list = sorted(feature_pvalue_list, key=lambda x: x[1], reverse=False)
        self.cols_sorted_with_pvalue = sorted_list
        return [x[0] for x in sorted_list]

    def analyze_pvalue(self, is_plot=True, top_features=10):
        from chester.util import ReportCollector, REPORT_PATH
        rc = ReportCollector(REPORT_PATH)
        self.sort_by_pvalue()
        rc.save_object(self.cols_sorted_with_pvalue[0:50],
                       text="top 50 with lowest partial pvalue for categorical feat based on chi square:")
        if len(self.cols) == 0:
            return None
        if is_plot:
            self.plot_wordcloud_pvalues(self.cols_sorted_with_pvalue)
            if self.n_cols > 50:
                self.plot_histogram_pvalues(self.cols_sorted_with_pvalue)
        print("Pvalues for top categorical features:")
        print(pd.DataFrame(self.cols_sorted_with_pvalue[0:top_features], columns=["feature", "pvalue"]))

    def partial_plot(self, classification_row_percent=True):
        import warnings
        warnings.simplefilter("ignore")
        top_features = 25
        if self.n_cols <= 25:
            sample_features = self.n_cols
            top_features = self.n_cols
        else:
            sample_features = min(50, int(self.n_cols / 2))

        top_feature_names = random.sample(self.cols_sorted[0:sample_features], top_features)
        feature_index = {feature: index for index, feature in enumerate(self.cols_sorted)}
        top_feature_names.sort(key=lambda x: feature_index[x])
        if self.data_info.problem_type_val in ["Binary regression", "Binary classification"]:
            if top_features == 1:
                fig, ax = plt.subplots(figsize=(14, 10))
                if classification_row_percent:
                    plt.suptitle("Partial Plot to Identify Patterns between Sampled Categorical Features and Target\n"
                                 "Showing % from Feature (row)",
                                 fontsize=14, fontweight='bold')
                else:
                    plt.suptitle("Partial Plot to Identify Patterns between Sampled Categorical Features and Target\n"
                                 "Showing % from Target (column)",
                                 fontsize=14, fontweight='bold')
                if classification_row_percent:
                    crosstab = pd.crosstab(self.data[top_feature_names[0]], self.target, normalize='index') * 100
                    crosstab = crosstab[(crosstab.T != 0).any()]
                    crosstab = crosstab.loc[:, (crosstab != 0).any(axis=0)]
                    crosstab = crosstab.loc[crosstab.sum(axis=1).sort_values(ascending=False).index[:5]]
                else:
                    crosstab = pd.crosstab(self.data[top_feature_names[0]], self.target, normalize='columns') * 100
                    crosstab = crosstab[(crosstab.T != 0).any()]
                    crosstab = crosstab.loc[:, (crosstab != 0).any(axis=0)]
                    crosstab = crosstab.loc[crosstab.sum(axis=1).sort_values(ascending=False).index[:5]]
                sns.heatmap(crosstab, annot=False, cmap="YlGnBu", fmt='g', ax=ax)
                ax.set_ylabel(None)
                ax.set_xlabel(None)
                ax.set_title(top_feature_names[0], fontsize=12, fontweight='bold')
                plt.show()
                plt.close()

            else:
                max_plots = min(9, top_features)
                dim = min(math.ceil(math.sqrt(max_plots)), 2)
                fig, ax = plt.subplots(dim, dim, figsize=(14, 4 + 4 * dim))
                fig.tight_layout()
                if classification_row_percent:
                    plt.suptitle("Partial Plot to Identify Patterns between Sampled Categorical Features and Target\n"
                                 "Showing % from Feature (row)",
                                 fontsize=14, fontweight='bold')
                else:
                    plt.suptitle("Partial Plot to Identify Patterns between Sampled Categorical Features and Target\n"
                                 "Showing % from Target (column)",
                                 fontsize=14, fontweight='bold')
                for i, col in enumerate(top_feature_names):
                    if i >= dim * dim:
                        break
                    ax_i = ax[i // dim, i % dim]
                    if classification_row_percent:
                        crosstab = pd.crosstab(self.data[col], self.target, normalize='index') * 100
                        crosstab = crosstab[(crosstab.T != 0).any()]
                        crosstab = crosstab.loc[:, (crosstab != 0).any(axis=0)]
                        crosstab = crosstab.loc[crosstab.sum(axis=1).sort_values(ascending=False).index[:5]]
                    else:
                        crosstab = pd.crosstab(self.data[col], self.target, normalize='columns') * 100
                        crosstab = crosstab[(crosstab.T != 0).any()]
                        crosstab = crosstab.loc[:, (crosstab != 0).any(axis=0)]
                        crosstab = crosstab.loc[crosstab.sum(axis=1).sort_values(ascending=False).index[:5]]
                    sns.heatmap(crosstab, annot=False, cmap="YlGnBu", fmt='g', ax=ax_i)
                    ax_i.set_ylabel(None)
                    ax_i.set_xlabel(None)
                    ax_i.set_title(col, fontsize=12, fontweight='bold')
                plt.tight_layout()
                plt.show()
                plt.close()
        if self.data_info.problem_type_val in ["Regression"]:
            max_plots = 9
            top_n = self.data[:top_features].columns
            dim = max(math.floor(math.sqrt(min(max_plots, len(top_n)))), 2)
            fig, ax = plt.subplots(dim, dim, figsize=(20, 4 + 4 * dim))
            target = self.target
            plt.suptitle(
                "Partial Plot to Identify Patterns between Categorical Sampled Features and Target (grouped by kmeans)",
                fontsize=16,
                fontweight='bold')
            grid_size = math.ceil(math.sqrt(top_features))
            num_features = min(grid_size * grid_size, top_features)
            for i, col in enumerate(top_feature_names[:num_features]):
                if i >= dim * dim:
                    break
                plt.subplot(dim, dim, i + 1)
                column = self.data[col].apply(lambda x: "'" + str(x))
                new_df = pd.concat([column, target], axis=1)
                top_values = list(column.value_counts().nlargest(5).index)
                data_filtered = new_df[column.isin(top_values)]
                sns.boxplot(x=col, y=target, data=data_filtered)
                plt.ylabel("Target")
                plt.xlabel("{} Value".format(col))
                plt.subplots_adjust(hspace=0.5, wspace=0.5)
            plt.show()
            plt.close()
        elif self.data_info.problem_type_val in ["Multiclass classification"]:
            max_plots = 16
            top_n = self.data[:max_plots].columns
            dim = max(math.floor(math.sqrt(len(top_n))), 2)
            fig, ax = plt.subplots(dim, dim, figsize=(19, 4 * dim))
            fig.tight_layout()
            plt.suptitle(
                "Heatmap to Show Correlation between Sampled Categorical Features (top 5 categories) and Target",
                fontsize=15,
                fontweight='bold')
            for i, col in enumerate(top_feature_names):
                if i >= dim * dim:
                    break
                plt.subplot(dim, dim, i + 1)
                crosstab = pd.crosstab(self.data[col], self.target, normalize='index') * 100
                crosstab = crosstab[(crosstab.T != 0).any()]
                crosstab = crosstab.loc[:, (crosstab != 0).any(axis=0)]
                crosstab = crosstab.loc[crosstab.sum(axis=1).sort_values(ascending=False).index[:5]]
                sns.heatmap(crosstab, annot=False, cmap="YlGnBu", fmt='g')
                plt.title(col, fontsize=10, fontweight='bold')
            plt.show()
            plt.close()

    @staticmethod
    def plot_histogram_pvalues(features_pvalues):
        """
        Plot histogram of p-values for features.
        :param features_pvalues: List of tuples (column name, pvalue).
        :return: None.
        """
        pvalues = [pvalue for _, pvalue in features_pvalues]
        fig, ax = plt.subplots()
        plt.figure(figsize=(15, 15))
        ax.hist(pvalues, bins=50, edgecolor='k', color='#2ecc71')
        ax.set_title("Histogram of P-values for Categorical Features", fontsize=16)
        ax.set_xlabel("P-value", fontsize=14)
        ax.set_ylabel("Frequency", fontsize=14)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.tick_params(axis='both', which='both', labelsize=12)
        plt.show()
        plt.close()

    @staticmethod
    def plot_wordcloud_pvalues(features_pvalues,
                               title="Categorical Features Pvalues Based on Partial Plot"):
        """
        Plot word cloud of features weighted by their p-value.
        :param features_pvalues: List of tuples (column name, pvalue).
        :param title: Title of the plot.
        :return: None.
        """
        features_pvalues = [(feature, 1 - pvalue) for feature, pvalue in features_pvalues]

        n_features = len(features_pvalues)
        width = min(int(900 * n_features / 4), 900)
        height = min(int(500 * n_features / 4), 500)
        wordcloud = WordCloud(width=width, height=height).generate_from_frequencies(dict(features_pvalues))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.title(title, fontsize=15)
        plt.show()
        plt.close()

    def run(self, is_plot=True):
        if self.n_cols > 1:
            self.analyze_pvalue(is_plot=is_plot)
            if is_plot:
                if 'classification' or 'binary regression' in self.data_info.problem_type_val.lower():
                    self.partial_plot(classification_row_percent=False)
                    self.partial_plot(classification_row_percent=True)
                else:
                    self.partial_plot()
                self.tsne()
        elif self.n_cols == 1:
            self.analyze_pvalue(is_plot=is_plot)
            if is_plot:
                self.partial_plot()
        plt.show()
        plt.close()


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
