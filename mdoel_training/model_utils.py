import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from mdoel_training.data_preparation import CVData
from model_analyzer.model_analysis import ModelAnalyzer


def organize_results(results):
    return pd.DataFrame(results)


def analyze_results(results, parameters):
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

    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    print(results_melt)

    # create boxplots to compare metrics across train and test and across different folds
    sns.boxplot(x='fold', y='value', hue='variable', data=results_melt, palette="Set3")
    plt.show()
