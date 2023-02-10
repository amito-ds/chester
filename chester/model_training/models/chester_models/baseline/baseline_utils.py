from chester.model_training.data_preparation import CVData
from chester.model_training.models.chester_models.base_model_utils import calculate_metrics_scores
from chester.model_training.models.chester_models.base_model_utils import is_metric_higher_is_better
from chester.model_training.models.chester_models.baseline.baseline_model import BaselineModel


def train_baseline(X_train, y_train,
                   baseline_value=None,
                   mode_baseline=False,
                   median_baseline=False,
                   avg_baseline=False):
    """
    Trains a baseline model using the given parameters.
    :param X_train: The training data features (unused in this function)
    :param y_train: The training data labels
    :param baseline_num: A number to use as the baseline (mean, median, mode, or percentile)
    :param percentile: Percentile to use as the baseline if baseline_num is not provided
    :return: A trained baseline model
    """
    model = BaselineModel(baseline_value=baseline_value, mode_baseline=mode_baseline,
                          median_baseline=median_baseline, avg_baseline=avg_baseline)
    model.fit(y_train)
    return model


def predict_baseline(model, X):
    """
    Makes predictions using a baseline model
    :param model: trained baseline model
    :param X: The data to make predictions on (unused in this function)
    :return: A list of predictions
    """
    return model.transform(X)


def baseline_with_outputs(cv_data: CVData,
                          metrics: list,
                          target_col: str,
                          baseline_value=None,
                          mode_baseline=False,
                          median_baseline=False,
                          avg_baseline=False,
                          ):
    results = []
    for i, (train_index, test_index) in enumerate(cv_data.splits):
        X_train, X_test = cv_data.train_data.iloc[train_index], cv_data.train_data.iloc[test_index]
        y_train, y_test = X_train[target_col], X_test[target_col]
        model = train_baseline(X_train, y_train, baseline_value=baseline_value, mode_baseline=mode_baseline,
                               median_baseline=median_baseline, avg_baseline=avg_baseline)
        prediction = predict_baseline(model, X_test)
        prediction_train = predict_baseline(model, X_train)
        scores = calculate_metrics_scores(y_test, prediction, metrics)
        results.append({'type': 'test', 'fold': i, **scores})
        scores = calculate_metrics_scores(y_train, prediction_train, metrics)
        results.append({'type': 'train', 'fold': i, **scores})
    return results, model


import pandas as pd


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
