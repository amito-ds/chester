import pandas as pd
from dateutil.parser import parse

from chester.zero_break.text_detector import determine_if_text_or_categorical_column


# need an extension for ts module
def is_date(string):
    try:
        parse(string)
        return True
    except:
        return False


class DataInfo:
    def __init__(self, data: pd.DataFrame, target: str = None):
        self.data = data
        self.target = target
        self.is_model = False if self.target is None else True
        self.problem_type_val = None
        self.feature_types_val = None
        self.loss_detector_val = None
        self.metrics_detector_val = None
        self.model_selection_val = None
        self.label_transformation_val = None
        self.rows = self.data.shape[0]

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
            if self.data[[self.target]].drop_duplicates().shape[0] == 2:
                return "Binary regression"
            else:
                return "Regression"
        elif self.data[[self.target]].drop_duplicates().shape[0] == 2:
            return "Binary classification"
        else:
            return "Multiclass classification"

    def _determine_time_cols(self):
        time_cols = []
        for col in self.data.columns:
            if self.data[col].dtype in ['int64', 'float64']:
                pass
            elif col == self.target:
                pass
            elif pd.api.types.is_datetime64_dtype(self.data[col]):
                time_cols.append(col)
            else:
                non_missing_values = self.data[col][self.data[col].notna()].astype(str)
                count = 0
                for value in non_missing_values:
                    if is_date(value):
                        count += 1
                if count / non_missing_values.shape[0] >= 0.9:
                    time_cols.append(col)
        return time_cols

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
        time_cols = self._determine_time_cols()
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

        for numeric_col in numerical_cols:
            if numeric_col in time_cols:
                time_cols.remove(numeric_col)
        for time_col in time_cols:
            if time_col in text_cols:
                text_cols.remove(time_col)
            elif time_col in categorical_cols:
                categorical_cols.remove(time_col)
        return {'numeric': numerical_cols, 'boolean': boolean_cols, 'text': text_cols,
                'categorical': categorical_cols, 'time': time_cols}

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
        # report += "Label Transformation: " + str(self.label_transformation_val)
        return report
