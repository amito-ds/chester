from typing import List

import numpy as np
from prettytable import PrettyTable
from sklearn.exceptions import UndefinedMetricWarning

from chester.feature_stats.utils import create_pretty_table
from chester.model_training.data_preparation import CVData
from chester.model_training.data_preparation import Parameter
from chester.model_training.models.chester_models.base_model_utils import is_metric_higher_is_better
from chester.model_training.models.chester_models.catboost.catboost_model import CatboostModel
from chester.model_training.models.chester_models.hp_generator import HPGenerator
from chester.zero_break.problem_specification import DataInfo


def train_catboost(X_train, y_train, parameters: list, data_info: DataInfo):
    """
    Trains a baseline model using the given parameters.
    :param X_train: The training data features (unused in this function)
    :param y_train: The training data labels
    :param parameters: parameters list
    :return: A trained baseline model
    """
    model = CatboostModel(parameters, data_info)
    model.fit(X_train, y_train)
    return model


def predict_catboost(model, X):
    """
    Makes predictions using a baseline model
    :param model: trained baseline model
    :param X: The data to make predictions on (unused in this function)
    :return: A list of predictions
    """
    return model.transform(X)


def catboost_with_outputs(cv_data: CVData,
                          metrics: list,
                          target_col: str,
                          parameters: list,
                          data_info: DataInfo,
                          ):
    results = []
    problem_type = data_info.problem_type_val
    for i, (X_train, y_train, X_test, y_test) in enumerate(cv_data.format_splits()):
        model = train_catboost(X_train, y_train, parameters=parameters, data_info=data_info)
        prediction = predict_catboost(model, X_test)
        prediction_train = predict_catboost(model, X_train)
        scores = calculate_catboost_metrics_scores(y_test, prediction, metrics, problem_type)
        results.append({'type': 'test', 'fold': i, **scores})
        scores = calculate_catboost_metrics_scores(y_train, prediction_train, metrics, problem_type)
        results.append({'type': 'train', 'fold': i, **scores})
    return results, model


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


default_parameters = {
    'iterations': 1000,
    'learning_rate': 0.03,
    'depth': 6,
    'random_seed': 42,
    'verbose': False
}

# For regression
regression_parameters = default_parameters.copy()
regression_parameters.update({'loss_function': 'RMSE', 'eval_metric': 'RMSE'})

# For binary classification
binary_classification_parameters = default_parameters.copy()
binary_classification_parameters.update({'loss_function': 'Logloss', 'eval_metric': 'AUC'})

# For multiclass classification
multiclass_classification_parameters = default_parameters.copy()
multiclass_classification_parameters.update({'loss_function': 'MultiClass', 'eval_metric': 'Accuracy'})


def generate_catboost_configs(k: int = 10) -> List[List[Parameter]]:
    hp_generator = HPGeneratoCatboost(n_models=k)
    parameter_format = hp_generator.hp_format(hp_generator.generate_configs())
    return parameter_format


class HPGeneratoCatboost(HPGenerator):
    def __init__(self, best_practice_configs: list = None,
                 categorical_sample_configs: list = None,
                 n_models=9,
                 best_practice_prob=0):
        super().__init__(best_practice_configs, categorical_sample_configs, n_models, best_practice_prob)
        self.best_practice_configs = []

    def generate_random_config(self) -> dict:
        config = {}
        # Sampling depth
        depth = np.random.randint(2, 6)
        config['depth'] = depth
        # Sampling learning rate
        learning_rate = np.random.uniform(0.01, 0.2)
        config['learning_rate'] = learning_rate
        # Sampling iterations
        iterations = np.random.randint(100, 1000)
        config['iterations'] = iterations
        # Sampling l2 regularization
        l2_reg = 10 ** np.random.uniform(-5, 5)
        config['l2_leaf_reg'] = l2_reg
        # Sampling random strength
        random_strength = np.random.uniform(1, 100)
        config['random_strength'] = random_strength
        # Sampling bagging temperature
        bagging_temp = np.random.uniform(0, 1)
        config['bagging_temperature'] = bagging_temp
        config['verbose'] = False
        return config


def calculate_catboost_metric_score(y_true, y_pred, metric, problem_type_val):
    import warnings
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    metric_name = metric.__name__
    if problem_type_val in ["Multiclass classification"]:
        y_pred = [item[0] for item in y_pred]
    try:
        return metric_name, metric(y_true, y_pred)
    except:
        try:
            if problem_type_val in ["Binary regression", "Binary classification"]:
                return metric_name, metric(y_true, y_pred, average='binary')
            elif problem_type_val == "Multiclass classification":
                return metric_name, metric(y_true, y_pred, average='macro')
        except:
            pass
    return None, None


def calculate_catboost_metrics_scores(y, prediction, metrics_list, problem_type=None):
    results = {}
    for metric in metrics_list:
        metric_name, metric_value = calculate_catboost_metric_score(y, prediction, metric, problem_type)
        results[metric_name] = metric_value
    return results
