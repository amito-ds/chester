from chester.feature_stats.utils import create_pretty_table
from chester.model_training.models.chester_models.base_model_utils import is_metric_higher_is_better

class_name_dict = {
    'LinearRegression': 'Linear Model',
    'LGBMRegressor': 'Light GBM',
    'LogisticRegression': 'Logistic Regression',
    'Sequential': 'Sequential Model',
    'RandomForestRegressor': 'Random Forest',
    'XGBRegressor': 'XGBoost',
    'SVR': 'Support Vector Regression',
    'KNeighborsRegressor': 'K-Neighbors Regression',
    'DecisionTreeRegressor': 'Decision Tree',
    'MLPRegressor': 'Multi-layer Perceptron',
    'AdaBoostRegressor': 'AdaBoost',
    'GradientBoostingRegressor': 'Gradient Boosting',
    'ExtraTreesRegressor': 'Extra Trees',
    'BaggingRegressor': 'Bagging',
    'CatBoostRegressor': 'CatBoost',
    'Lasso': 'Lasso',
    'Ridge': 'Ridge',
    'ElasticNet': 'Elastic Net',
    'PassiveAggressiveRegressor': 'Passive Aggressive'
}


def get_model_name(model):
    model_name = class_name_dict.get(model.__class__.__name__, model.__class__.__name__)
    return model_name


def compare_models(results):
    all_results = [(pd.DataFrame(result), model) for result, model in results]
    metric_name = [col for col in all_results[0][0].columns if col not in ['type', 'fold']][0]
    sort_ascending = is_metric_higher_is_better(metric_name)
    best_result = None
    best_model = None
    best_value = None
    for (result, model) in all_results:
        test_result = result[result['type'] == 'test'].groupby('fold').mean(numeric_only=True).reset_index()
        mean_value = test_result[metric_name].mean()
        if best_value is None or \
                (sort_ascending and mean_value > best_value) \
                or (not sort_ascending and mean_value < best_value):
            best_value = mean_value
            best_result = result
            best_model = model
    return best_result, best_model


def calculate_average(df):
    df_grouped = df.drop(columns=['fold']).groupby(['type', 'model'], as_index=False).mean(numeric_only=True)
    return df_grouped


import pandas as pd
import matplotlib.pyplot as plt


def visualize_performance(df, with_baseline=True):
    addition = 'excluding baseline model' if not with_baseline else ''

    # Create pivot table to summarize mean performance metric by type and model
    metric_columns = [col for col in df.columns if col not in ["type", "model"]]
    metric_columns = [metric for metric in metric_columns if metric is not None]
    pivot = df.pivot(index="model", columns="type", values=metric_columns)

    if not with_baseline:
        pivot = df[df['model'] != 'BaselineModel'].pivot(index="model", columns="type", values=metric_columns)

    # Plot bar chart to compare mean performance metric by model
    # fig, axes = plt.subplots(nrows=len(metric_columns), ncols=1, figsize=(12, 10), sharex=True)
    fig, axes = plt.subplots(nrows=len(metric_columns), ncols=1, figsize=(12, 10), sharex=True)

    for i, metric in enumerate(metric_columns):
        pivot[metric].plot.bar(ax=axes[i], rot=0)
        axes[i].set_ylabel(None)
        axes[i].set_title("Comparison of {} by Model {}".format(metric, addition))

    plt.xlabel(None)
    plt.ylabel(None)
    plt.tight_layout()
    plt.show()
    plt.close()

    return pivot


def compare_best_models(results, plot_results=True):
    from chester.util import ReportCollector, REPORT_PATH
    rc = ReportCollector(REPORT_PATH)

    all_results = []
    all_results_with_models = []
    for res in results:
        result, model = res
        result_organized = pd.DataFrame(result)
        result_organized['model'] = get_model_name(model)
        all_results.append(result_organized)
        all_results_with_models.append((result_organized, model))

    metrics_results = calculate_average(pd.concat(all_results))
    if plot_results:
        print("Model results - comparing the best out of each type")
        visualize_performance(metrics_results, with_baseline=True)
        print("Model results - comparing the best out of each type, excluding baseline model")
        visualize_performance(metrics_results, with_baseline=False)

    rc.save_object(create_pretty_table(metrics_results),
                   text="Model results compared - showing the best out of each type after CV & HP tuning: ")

    metric_name = [col for col in all_results[0].columns if col not in ['type', 'fold', 'model']][0]
    sort_ascending = is_metric_higher_is_better(metric_name)
    best_result = None
    best_model = None
    best_value = None
    for (result, model) in all_results_with_models:
        test_result = result[result['type'] == 'test'].groupby('fold').mean(numeric_only=True).reset_index()
        mean_value = test_result[metric_name].mean()
        if best_value is None or \
                (sort_ascending and mean_value > best_value) \
                or (not sort_ascending and mean_value < best_value):
            best_value = mean_value
            best_result = result
            best_model = model
    print(f"Optimized {metric_name}, with best value: {round(best_value, 2)}. ")
    return best_result, best_model
