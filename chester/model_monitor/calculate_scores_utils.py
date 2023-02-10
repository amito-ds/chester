def calculate_metric_score(y_true, y_pred, metric, problem_type_val):
    metric_name = metric.__name__
    try:
        # print("trying to", metric(y_true, y_pred))
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
    results = {}
    for metric in metrics_list:
        metric_name, metric_value = calculate_metric_score(y, prediction, metric, problem_type)
        results[metric_name] = metric_value
    return results
