from sklearn.preprocessing import LabelEncoder


def calculate_metric_score(y_true, y_pred, metric):
    metric_name = metric.__name__
    try:
        return metric_name, metric(y_true, y_pred)
    except ValueError as e:
        # Convert string labels to integers
        le = LabelEncoder()
        y_true = le.fit_transform(y_true)
        y_pred = le.transform(y_pred)
        return metric_name, metric(y_true, y_pred)


def calculate_metrics_scores(y, prediction, metrics_list):
    results = {}
    for metric in metrics_list:
        metric_name, metric_value = calculate_metric_score(y, prediction, metric)
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
