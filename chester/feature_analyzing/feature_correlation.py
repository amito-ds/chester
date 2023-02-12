import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency

tsne_plot_message = """
The t-SNE plot is a visualization tool that is used to reduce the dimensionality of the data 
and visualize the relationships between the features and the target label in a 2D space. 
The plot shows the transformed data points, where each point represents a sample in the dataset 
and its color represents the target label. The plot can help you understand 
the relationships between the features and the target label in a more intuitive way.
"""

correlation_matrix_message = """
The correlation matrix is a tool used to measure the strength and direction of the linear relationship 
between different features in the dataset. The correlation coefficient ranges from -1 to 1, where a value 
of 1 indicates a perfect positive correlation, meaning that as one feature increases, the other feature 
also increases, a value of -1 indicates a perfect negative correlation, meaning that as one feature 
increases, the other feature decreases, and a value of 0 indicates no correlation between the features. 
It's important to note that a high correlation does not necessarily imply causality, it just indicates 
that the two features are related. In this report, we present the correlation matrix of the features 
in the dataset using a heatmap, where a darker color indicates a stronger correlation. It's worth noting 
that the correlation between the feature and target column (if provided) is also presented. In general, 
a correlation coefficient of 0.7 or higher is considered a strong correlation, a coefficient between 
0.3 and 0.7 is considered a moderate correlation, and a coefficient below 0.3 is considered a weak correlation. 
However, it's important to consider the context of the problem and the domain knowledge when interpreting 
the correlation matrix.
"""


class PreModelAnalysis:
    def __init__(self,
                 df: pd.DataFrame = None,
                 target_column: str = 'target',
                 top_n_features: int = 200,
                 correlation_matrix=True,
                 tsne_plot=True,
                 top_n_pairplot=False,
                 chi_square_test_all_features=True
                 ):
        self.df = df
        self.top_n_features = top_n_features
        self.target_column = target_column
        self.is_model = not (not target_column)
        self.correlation_matrix_bool = correlation_matrix
        self.tsne_plot_bool = tsne_plot
        self.top_n_pairplot_bool = top_n_pairplot
        self.chi_square_test_all_features_bool = chi_square_test_all_features

    def generate_report(self):
        report_str = ""
        if self.correlation_matrix_bool:
            report_str += "Generating Correlation Matrix, "
        if self.tsne_plot_bool:
            report_str += "Generating t-SNE Plot, "
        if self.top_n_pairplot_bool:
            report_str += f"Generating top {self.top_n_features} features Pairplot, "
        if self.chi_square_test_all_features_bool:
            report_str += "Running Chi-Square test on all features, "
        if report_str:
            report_str = report_str[:-2]
            print(f"The following EDA steps will be applied: {report_str}.")
        else:
            print("No EDA steps selected.")

    def get_model_type(self, class_threshold: int = 2):
        is_classification, is_regression = (False, False)
        if not self.is_model:
            return is_classification, is_regression
        target_values = self.df[self.target_column]
        if target_values.dtype == object:
            is_classification = True
        elif target_values.nunique() <= class_threshold:
            is_classification, is_regression = (True, True)
        else:
            is_regression = True
        return is_classification, is_regression

    def correlation_matrix(self):
        print(correlation_matrix_message)
        corr = self.df.corr()
        plt.figure(figsize=(10, 8))
        plt.title("Feature correlation matrix")
        sns.heatmap(corr, annot=True)
        plt.show()
        plt.close()
        return corr

    def tsne_plot(self, n_components=2, perplexity=30.0, n_iter=1000):
        print(tsne_plot_message)
        from sklearn.manifold import TSNE
        X = self.df.drop(columns=self.target_column)
        y = self.df[self.target_column]
        tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter)
        X_tsne = tsne.fit_transform(X)
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
        plt.title("t-SNE Plot of Features and Target Label")
        plt.show()
        plt.close()

    def top_n_pairplot(self, N=4, trimming_left=0.05, trimming_right=0.05):
        if trimming_left > 0:
            print(f"left trimming {100 * trimming_left}% ")
        if trimming_right > 0:
            print(f"right trimming {100 * trimming_right}% ")
        import seaborn as sns
        corr = self.df.corr()
        top_n_features = corr.nlargest(N, self.target_column).index
        top_n_features = top_n_features.drop(self.target_column)
        X = self.df[top_n_features]
        X = X.dropna()
        X = X[(X > X.quantile(trimming_left)) & (X < X.quantile(1 - trimming_right))]
        sns.pairplot(X)
        plt.show()
        plt.close()

    def chi_square_test(self, feature):
        from scipy.stats import chi2_contingency
        X = self.df[[feature, self.target_column]]
        X = X.dropna()
        crosstab = pd.crosstab(X[feature], X[self.target_column])
        chi2, p, dof, expected = chi2_contingency(crosstab)
        return p

    def chi_square_test_all_features(self, k=3):
        pd.options.mode.chained_assignment = None
        from sklearn.cluster import KMeans
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)

        pvalues = {}
        for feature in self.df.columns:
            if feature != self.target_column:
                X = self.df[[feature]]
                kmeans = KMeans(n_clusters=k, n_init=10)
                kmeans.fit(X)
                X[feature + "_cluster"] = kmeans.labels_
                crosstab = pd.crosstab(X[feature + "_cluster"], self.df[self.target_column])
                chi2, p, dof, expected = chi2_contingency(crosstab)
                pvalues[feature] = p
        return pvalues

    def plot_pvalues(self, threshold=20):
        if not self.is_model:
            pass
        pvalues = self.chi_square_test_all_features()
        if len(pvalues) < threshold:
            print(f"Number of features is less than {threshold}. Not enough data for histogram plot.")
            pass
        import matplotlib.pyplot as plt
        plt.hist(list(pvalues.values()), bins=20)
        plt.xlabel("p-values")
        plt.ylabel("Frequency")
        plt.title("Histogram of p-values")
        plt.show()
        plt.close()

    def run(self):
        import subprocess
        if self.top_n_features:
            self.df = self.df[self.select_top_variance_features(self.top_n_features)]
        if self.correlation_matrix_bool:
            self.correlation_matrix()
        if self.tsne_plot_bool:
            try:
                subprocess.run(["python", "tsne_plot.py"], timeout=120)
            except subprocess.TimeoutExpired:
                print("t-SNE plot did not complete within 2 minutes.")
        if self.top_n_pairplot_bool:
            self.top_n_pairplot()
        if self.chi_square_test_all_features_bool:
            self.plot_pvalues()

    def select_top_variance_features(self, n=200):
        variances = self.df.var()
        top_features = variances.sort_values(ascending=False).head(n).index
        return top_features
