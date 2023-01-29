from typing import Dict, List

import pandas as pd

from chester.zero_break.text_detector import determine_if_text_or_categorical_column


class DataInfo:
    def __init__(self, data: pd.DataFrame, target: str = None):
        self.data = data
        self.target = target
        self.problem_type_val = None
        self.feature_types_val = None
        self.loss_detector_val = None
        self.metrics_detector_val = None
        self.model_selection_val = None
        self.label_transformation_val = None

    def calculate(self):
        self.problem_type_val = self.problem_type()
        self.feature_types_val = self.feature_types()
        self.loss_detector_val = self.loss_detector()
        self.metrics_detector_val = self.metrics_detector()
        self.model_selection_val = self.model_selection()
        self.label_transformation_val = self.label_transformation()

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

    def _determine_boolean_cols(self):
        boolean_cols = []
        for col in self.data.columns:
            if pd.api.types.is_bool_dtype(self.data[col]):
                boolean_cols.append(col)
        return boolean_cols

    def feature_types(self):
        numerical_cols = self._determine_numerical_cols()
        boolean_cols = self._determine_boolean_cols()
        text_cols = []
        categorical_cols = []
        for col in self.data.columns:
            if col not in numerical_cols and col not in boolean_cols:
                is_text, is_categorical = determine_if_text_or_categorical_column(self.data[col])
                if is_text:
                    text_cols.append(col)
                elif is_categorical:
                    categorical_cols.append(col)
        if self.target in numerical_cols:
            numerical_cols.remove(self.target)
        if self.target in boolean_cols:
            boolean_cols.remove(self.target)
        if self.target in text_cols:
            text_cols.remove(self.target)
        if self.target in categorical_cols:
            categorical_cols.remove(self.target)
        return {'numeric': numerical_cols, 'boolean': boolean_cols, 'text': text_cols, 'categorical': categorical_cols}

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
            return {"linear", "logistic", "catboost", "baseline-median", "baseline-average"}
        elif problem_type == "Regression":
            return {"linear", "catboost", "baseline-median", "baseline-average"}
        elif problem_type == "Binary classification":
            return {"logistic", "catboost", "baseline-mode"}
        elif problem_type == "Multiclass classification":
            return {"logistic", "catboost", "baseline-mode"}

    def label_transformation(self):
        if self.problem_type() == "No target variable":
            return None
        elif self.problem_type() != "Regression":
            return None
        else:
            return ["Winsorizing", "Log Transformation", "Square Root Transformation", "Reciprocal transformation"]

    def __str__(self):
        report = "Data Information Report\n"
        if self.problem_type_val:
            report += "Problem Type: " + self.problem_type_val + "\n"
        if self.target:
            report += "Target Variable: " + self.target + "\n"
        report += "Feature Types: " + str(self.feature_types_val) + "\n"
        if self.target:
            report += "Loss Function: " + self.loss_detector_val + "\n"
        report += "Evaluation Metrics: " + str(self.metrics_detector_val) + "\n"
        report += "Model Selection: " + str(self.model_selection_val) + "\n"
        report += "Label Transformation: " + str(self.label_transformation_val)
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


# Create a DataFrame with 30 columns
# data = pd.DataFrame({'target': [1, 2, 3, 4],
#                     'a': [1, 2, 3, 4],
#                     'b': [5, 6, 7, 8],
#                     'c': ['a', 'b', 'c', 'd'],
#                     'd': [1.1, 2.2, 3.3, 4.4],
#                     'e': [True, False, True, False],
#                     'f': [1, 2, 3, 4],
#                     'g': [5, 6, 7, 8],
#                     'h': ['a', 'b', 'c', 'd'],
#                     'i': [1.1, 2.2, 3.3, 4.4],
#                     'j': [True, False, True, False],
#                     'k': [1, 2, 3, 4],
#                     'l': [5, 6, 7, 8],
#                     'm': ['a', 'b', 'c', 'd'],
#                     'n': [1.1, 2.2, 3.3, 4.4],
#                     'o': [True, False, True, False],
#                     'p': [1, 2, 3, 4],
#                     'q': [5, 6, 7, 8],
#                     'r': ['a', 'b', 'c', 'd'],
#                     's': [1.1, 2.2, 3.3, 4.4],
#                     't': [True, False, True, False],
#                     'u': [1, 2, 3, 4],
#                     'v': [5, 6, 7, 8],
#                     'w': ['a', 'b', 'c', 'd'],
#                     'x': [1.1, 2.2, 3.3, 4.4],
#                     'y': [True, False, True, False],
#                     'z': [1, 2, 3, 4]
#                    })

from data_loader.webtext_data import load_data_pirates, load_data_king_arthur
from chester.features_engineering.fe_nlp import FeatureExtraction

from chester.run_full_cycle import run_tcap, DataSpec

df1 = load_data_pirates().assign(target='chat_logs')
df2 = load_data_king_arthur().assign(target='pirates')
df = pd.concat([df1, df2])

# Create an instance of the DataInfo class
spec = DataInfo(df, target='target')

# Print the summary of the DataFrame
# calc = spec.calculate()
# print(spec)
