from typing import Dict, List

import pandas as pd

from chester.zero_break.text_detector import determine_if_text_or_categorical_column


class DataInfo:
    def __init__(self, data: pd.DataFrame, target: str = None):
        self.data = data
        self.target = target

    def has_target(self) -> bool:
        return self.target is not None

    def problem_type(self):
        if self.target is None:
            return "No target variable"
        elif pd.api.types.is_numeric_dtype(self.data[self.target]):
            if len(self.data[self.target].unique()) == 2:
                return "Binary regression"
            else:
                return "Regression"
        elif len(self.data[self.target].unique()) == 2:
            return "Binary classification"
        else:
            return "Multiclass classification"

    def _determine_numerical_cols(self):
        numerical_cols = []
        for col in self.data.columns:
            if self.data[col].dtype in ['int64', 'float64']:
                numerical_cols.append(col)
        return numerical_cols

    def feature_types(self):
        numerical_cols = self._determine_numerical_cols()
        text_cols = []
        categorical_cols = []
        for col in self.data.columns:
            if col not in numerical_cols:
                is_text, is_categorical = determine_if_text_or_categorical_column(self.data[col])
                if is_text:
                    text_cols.append(col)
                elif is_categorical:
                    categorical_cols.append(col)
        return {'numeric': numerical_cols, 'text': text_cols, 'categorical': categorical_cols}

    def loss_detector(self):
        problem_type = self.problem_type()
        if problem_type == "Regression":
            return "R squared"
        elif problem_type in ["Binary regression", "Binary classification", "Multiclass classification"]:
            return "Cross entropy"
        else:
            return None

    def metrics_detector(self):
        problem_type = self.problem_type()
        if problem_type == "No target variable":
            return None
        elif problem_type == "Binary regression":
            return ["MSE", "MAE", "MAPE", "ROC"]
        elif problem_type == "Binary classification":
            return ["Accuracy", "Precision", "Recall", "F1"]
        elif problem_type == "Regression":
            return ["MSE", "MAE", "MAPE"]
        elif problem_type == "Multiclass classification":
            return ["Accuracy", "Precision", "Recall", "F1"]

    def model_selection(self):
        problem_type = self.problem_type()
        if problem_type == "No target variable":
            return None
        elif problem_type == "Binary regression":
            return {"linear", "logistic", "catboost", "baseline - average"}
        elif problem_type == "Regression":
            return {"linear", "catboost", "baseline - average"}
        elif problem_type == "Binary classification":
            return {"logistic", "catboost", "baseline - mode"}
        elif problem_type == "Multiclass classification":
            return {"logistic", "catboost", "baseline - mode"}

    def label_transformation(self):
        if self.problem_type() == "No target variable":
            return None
        elif self.problem_type() != "Regression":
            return None
        else:
            return ["Winsorizing", "Log Transformation", "Square Root Transformation", "Reciprocal transformation"]

    def __str__(self):
        report = "Data Information Report\n"
        report += "Problem Type: " + self.problem_type() + "\n"
        if self.target:
            report += "Target Variable: " + self.target + "\n"
        report += "Feature Types: " + str(self.feature_types()) + "\n"
        if self.target:
            report += "Loss Function: " + self.loss_detector() + "\n"
            report += "Evaluation Metrics: " + str(self.metrics_detector()) + "\n"
            report += "Model Selection: " + str(self.model_selection()) + "\n"
            report += "Label Transformation: " + str(self.label_transformation())
        return report


# data = pd.DataFrame({'target': [1, 2, 3, 4, 5], 'feature1': [1, 2, 3, 4, 5], 'feature2': [2, 3, 4, 5, 6]})
# spec = DataInfo(data, target='target')
# print(spec)
#
#
# data = pd.DataFrame({'target': ['yes', 'no', 'yes', 'yes', 'no'], 'feature1': [1, 2, 3, 4, 5], 'feature2': [2, 3, 4, 5, 6]})
# spec = DataInfo(data, target='target')
# print(spec)
#
# data = pd.DataFrame({'target': ['apple', 'orange', 'banana', 'banana', 'apple'], 'feature1': [1, 2, 3, 4, 5], 'feature2': [2, 3, 4, 5, 6]})
# spec = DataInfo(data, target='target')
# print(spec)

# data = pd.DataFrame({'feature1': [1, 2, 3, 4, 5], 'feature2': [2, 3, 4, 5, 6]})
# spec = DataInfo(data)
# print(spec)
#
# # Binary classification example
# data = pd.DataFrame({'target': [0, 1, 0, 1], 'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8]})
# spec = DataInfo(data, target='target')
# print(spec)
#
# # Multiclass classification example
# data = pd.DataFrame({'target': ['a', 'b', 'c', 'a'], 'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8]})
# spec = DataInfo(data, target='target')
# print(spec)
#
# # Regression example
# data = pd.DataFrame({'target': [1, 2, 3, 4], 'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8]})
# spec = DataInfo(data, target='target')
# print(spec)
#
# # No target variable example
# data = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8]})
# spec = DataInfo(data)
# print(spec)
