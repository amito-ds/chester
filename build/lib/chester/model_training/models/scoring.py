import numpy as np


def calculate_score_model(y, prediction):
    """
    Calculates evaluation metrics for the predictions
    :param y: The true labels
    :param prediction: predictions
    :return: A dictionary of metric scores for each model
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import warnings
    warnings.filterwarnings('ignore')
    unique_values = len(np.unique(y))
    scores = {}
    if unique_values == 2:
        scores['accuracy_score'] = accuracy_score(y, prediction)
        scores['precision_score'] = precision_score(y, prediction)
        scores['precision_score'] = precision_score(y, prediction, zero_division=0)
        scores['recall_score'] = recall_score(y, prediction)
        scores['f1_score'] = f1_score(y, prediction)
    else:
        scores['accuracy_score'] = accuracy_score(y, prediction)
        scores['precision_score'] = precision_score(y, prediction, average='micro', zero_division=0)
        scores['recall_score'] = recall_score(y, prediction, average='micro')
        scores['f1_score'] = f1_score(y, prediction, average='micro')
    return scores
