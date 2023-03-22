import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning


def calculate_metric_score(y_true, y_pred, metric, problem_type_val):
    metric_name = metric.__name__
    if problem_type_val in ["Binary regression"]:
        y_pred = pd.Series(y_pred[:, 1], name='y_pred')
    elif problem_type_val in ["Multiclass classification", "Binary classification"]:
        y_pred = pd.Series(y_pred.argmax(axis=1), name='y_pred')
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


def calculate_metrics_scores(y, prediction, metrics_list, problem_type=None):
    import warnings
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    results = {}
    for metric in metrics_list:
        metric_name, metric_value = calculate_metric_score(y, prediction, metric, problem_type)
        results[metric_name] = metric_value
    return results


from fuzzywuzzy import process


def is_metric_higher_is_better(metric_name: str) -> bool:
    higher_better_metrics = ["accuracy", "f1", "precision", "recall", "roc", "roc auc", "gini", "r squared", "mape",
                             "mae", "mse", "accuracy_score", "f1_score", "precision_score", "recall_score", "roc_score",
                             "roc_auc_score"]
    lower_better_metrics = ["rmse", "log loss", "cross entropy", "brier score", "loss", "mean_squared_error",
                            "mean_absolute_error", "mean_absolute_percentage_error"]

    match = process.extractOne(metric_name, higher_better_metrics + lower_better_metrics)
    if match[1] > 90:
        if match[0] in higher_better_metrics:
            return True
        elif match[0] in lower_better_metrics:
            return False
    else:
        raise ValueError(f"Metric {metric_name} not recognized.")
