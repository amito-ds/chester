from collections import defaultdict
from typing import List
import pandas as pd
import seaborn as sns
from sklearn.exceptions import UndefinedMetricWarning

from chester.model_analyzer.model_analysis import get_traffic_light
# from chester.model_compare.compare_messages import get_model_name
from chester.model_training.data_preparation import CVData
from chester.model_training.data_preparation import Parameter
from chester.model_training.models.chester_models.base_model_utils import is_metric_higher_is_better
from chester.model_training.models.chester_models.catboost.catboost_model import CatboostModel


def train_catboost(X_train, y_train, parameters: list, problem_type: str):
    """
    Trains a baseline model using the given parameters.
    :param X_train: The training data features (unused in this function)
    :param y_train: The training data labels
    :param parameters: parameters list
    :return: A trained baseline model
    """
    model = CatboostModel(parameters, problem_type)
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
                          problem_type=None,
                          ):
    results = []
    for i, (X_train, y_train, X_test, y_test) in enumerate(cv_data.format_splits()):
        model = train_catboost(X_train, y_train, parameters=parameters, problem_type=problem_type)
        prediction = predict_catboost(model, X_test)
        prediction_train = predict_catboost(model, X_train)
        # print("prediction_train", prediction_train)
        # print(metrics)
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
    # print("all_results", all_results[0][0])
    metric_name = [col for col in all_results[0][0].columns if col not in ['type', 'fold']][0]
    sort_ascending = is_metric_higher_is_better(metric_name)
    best_result = None
    best_model = None
    best_value = None
    for (result, model) in all_results:
        # print("this is the results!", result)
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
    fig, axes = plt.subplots(nrows=len(metric_columns), ncols=1, figsize=(8, 8), sharex=True)
    for i, metric in enumerate(metric_columns):
        pivot[metric].plot.bar(ax=axes[i], rot=0)
        axes[i].set_ylabel(metric)
        axes[i].set_title("Comparison of {} by Model {}".format(metric, addition))

    plt.xlabel("Model")
    plt.tight_layout()
    plt.show()

    return pivot


def compare_best_models(results, plot_results=False):
    print(f"res len {len(results)}")
    all_results = []
    all_results_with_models = []
    for res in results:
        result, model = res
        # print(result)
        result_organized = pd.DataFrame(result)
        result_organized['model'] = get_model_name(model)
        all_results.append(result_organized)
        all_results_with_models.append((result_organized, model))
    if plot_results:
        visualize_performance(calculate_average(pd.concat(all_results)), with_baseline=True)
        visualize_performance(calculate_average(pd.concat(all_results)), with_baseline=False)

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
    print(f"Optimized {metric_name}, with best value: {best_value}. "
          f"Traffic light {get_traffic_light(metric_name, best_value)}")
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


def generate_catboost_configs(k: int, problem_type: str) -> List[List[Parameter]]:
    # default_parameters should be by problem type
    catboost_default_parameters = default_parameters
    if problem_type == "Binary regression" or problem_type == "Regression":
        catboost_default_parameters = regression_parameters
    elif problem_type == "Binary classification":
        catboost_default_parameters = binary_classification_parameters
    elif problem_type == "Multiclass classification":
        catboost_default_parameters = multiclass_classification_parameters

    # List of additional configurations to test
    additional_confs = [
        {**catboost_default_parameters, 'iterations': 100, 'depth': 4},
        {**catboost_default_parameters, 'iterations': 200, 'depth': 8},
        {**catboost_default_parameters, 'iterations': 300, 'depth': 6},
        {**catboost_default_parameters, 'iterations': 200, 'depth': 5},
        {**catboost_default_parameters, 'iterations': 100, 'depth': 7},
        {**catboost_default_parameters, 'iterations': 1000, 'depth': 4},
        {**catboost_default_parameters, 'iterations': 200, 'depth': 8},
        {**catboost_default_parameters, 'iterations': 300, 'depth': 5},
        {**catboost_default_parameters, 'iterations': 200, 'depth': 7},
        {**catboost_default_parameters, 'iterations': 100, 'depth': 6},
        {**catboost_default_parameters, 'iterations': 50, 'depth': 5},
        {**catboost_default_parameters, 'iterations': 50, 'depth': 7},
        {**catboost_default_parameters, 'iterations': 100, 'depth': 6},
        {**catboost_default_parameters, 'iterations': 200, 'depth': 8},
        {**catboost_default_parameters, 'iterations': 400, 'depth': 5},
        {**catboost_default_parameters, 'iterations': 600, 'depth': 7},
        {**catboost_default_parameters, 'iterations': 450, 'depth': 6},
        {**catboost_default_parameters, 'iterations': 200, 'depth': 8},
        {**catboost_default_parameters, 'iterations': 100, 'depth': 5}
    ]

    # List to store the final configurations
    catboost_parameters = []
    for conf in additional_confs[:k]:
        # Create a dictionary to store the final configuration
        final_conf = defaultdict()
        final_conf.update(conf)
        # Convert the dictionary to a list of Parameter objects
        final_conf = [Parameter(key, value) for key, value in final_conf.items()]
        catboost_parameters.append(final_conf)
    return catboost_parameters


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
