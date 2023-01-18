import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from cleaning.cleaning import clean_text
from data_loader.webtext_data import load_data_pirates, load_data_king_arthur
from features_engineering.fe_main import get_embeddings
from mdoel_training.best_model import ModelCycle
from mdoel_training.data_preparation import CVData
from model_analyzer.model_analysis import analyze_model
from preprocessing.preprocessing import preprocess_text, get_stemmer
from util import get_stopwords

# classification
# buckets + chi square

# regression
# feautre corrleation to the label

# both
# feature matrix correlation
# anomaly detection

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
    def __init__(self, df: pd.DataFrame, target_column: str = 200, top_n_features: int = 200):
        self.df = df
        self.target_column = target_column
        self.is_model = not (not target_column)
        if top_n_features:
            self.df = self.df[self.select_top_variance_features(top_n_features)]
        if target_column:
            self.df[target_column] = df[target_column]

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
                kmeans = KMeans(n_clusters=k)
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

    def feature_correlation(self):
        print("performing feature correlation")
        # get model type (classification, regression, both)
        # get if its a model

        # if not a model -> feature correlation (all features are numerics)
        # else:
        # if regression: plot correlation table, heatmap. last column is the target. choose the right correlation metric for this problem
        # if calssification:plot correlation table, heatmap. last column is the target. choose the right correlation metric for this problem

    def run(self, correlation_matrix=True, tsne_plot=True, top_n_pairplot=True, chi_square_test_all_features=True):
        if correlation_matrix:
            self.correlation_matrix()
        if tsne_plot:
            self.tsne_plot()
        if top_n_pairplot:
            self.top_n_pairplot()
        if chi_square_test_all_features:
            self.plot_pvalues()

    def select_top_variance_features(self, n=200):
        variances = self.df.var()
        top_features = variances.sort_values(ascending=False).head(n).index
        return top_features


# train_embedding = pd.read_csv("train_embedding.csv")
# test_embedding = pd.read_csv("test_embedding.csv")
# train_embedding = train_embedding.drop(train_embedding.columns[0], axis=1)
# test_embedding = test_embedding.drop(test_embedding.columns[0], axis=1)
df1 = load_data_pirates().assign(target='chat_logs')
df2 = load_data_king_arthur().assign(target='pirates')
df = pd.concat([df1, df2])
#
# # # Clean the text column
get_sw = get_stopwords()
df['text'] = df['text'].apply(lambda x: clean_text(x,
                                                   remove_stopwords_flag=True,
                                                   stopwords=get_sw))
#
# # preprocess the text column
df['clean_text'] = df['text'].apply(lambda x:
                                    preprocess_text(x, stemmer=get_stemmer('porter'), stem_flag=True))
#
# #
train_embedding, test_embedding = get_embeddings(training_data=df, corex=True, tfidf=True, bow=True, corex_dim=50)

#
# print(train_embedding.shape)
# print(test_embedding.shape)
#
target_col = 'target'
label_encoder = LabelEncoder()
train_embedding[target_col] = label_encoder.fit_transform(train_embedding[target_col])
test_embedding[target_col] = label_encoder.transform(test_embedding[target_col])

# print(train_embedding.shape[1])

pma = PreModelAnalysis(train_embedding, target_column=target_col)
print(pma.is_model)
print(pma.get_model_type())
# cor = pma.correlation_matrix()
# pma.tsne_plot(n_components=2)
# pma.top_n_pairplot(N=5, trimming_right=0.2, trimming_left=0)
# print(pma.plot_pvalues(threshold=5))
print(pma.run())
