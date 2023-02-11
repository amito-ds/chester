import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from chester.model_analyzer import model_analysis

analysis_message = "The following graph displays boxplots of the calculated metrics for each fold\n, " \
                   "with the x-axis representing the fold number and the y-axis representing the metric value.\n " \
                   "The boxplot shows the median, interquartile range, and outliers of the metric values \n for both " \
                   "train and test data\n. " \
                   "A large difference between the train and test boxplots may indicate overfitting."


def analyze_results(results: pd.DataFrame, parameters):
    # remove parameters columns from the results dataframe
    results_copy = results.drop([p.name for p in parameters], axis=1)
    # check if there are any lists in the dataframe
    if any(isinstance(i, list) for i in results_copy.values):
        # iterate through the dataframe and flatten any lists
        results_copy = results_copy.applymap(lambda x: x[0] if isinstance(x, list) else x)
    # keep only numeric columns and id_vars=['type', 'fold']
    results_copy = pd.concat([
        results_copy[['type', 'fold']]
        , results_copy.select_dtypes(include=['float64', 'int64'])]
        , axis=1)
    # melt the dataframe to have (type, fold, metric, value) format
    results_copy = results_copy.loc[:, ~results_copy.columns.duplicated()]
    results_melt = pd.melt(results_copy, id_vars=['type', 'fold'], value_vars=results_copy.columns[2:])
    results_melt = results_melt.astype({'fold': 'int64'})
    results_agg = results_melt.groupby(['type', 'variable']).agg({'value': 'mean'}).reset_index()
    results_pivot = results_agg.pivot(index='type', columns='variable', values='value')
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    print("Model performance:")
    print(model_analysis.AnalyzeMessages().performance_metrics_message())
    print(results_pivot)

    sns.boxplot(x='fold', y='value', hue='variable', data=results_melt, palette="Set3")
    print(analysis_message)
    plt.show()
    plt.close()
