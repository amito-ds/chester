import numpy as np
import pandas as pd

from tamtam.user_class.user_class import ABData, TestInfo
from tamtam.utils.column_utils import Columns


class ABInfo:
    def __init__(self, ab_data: ABData, test_info: TestInfo):
        self.ab_data = ab_data
        self.test_info = test_info
        self.df = self.ab_data.df
        self.run()  # validate cols
        # additional columns to calculate
        self.control_name = "A"
        self.treatment_name = "B"
        self.trimmed, self.trimmed_cols, self.trimmed_col_by_metric = self.trim()

    def add_dummies(self):
        if not self.test_info.is_weight_col():
            self.df[Columns.weight] = 1
        if not self.test_info.is_id_cols():
            self.df[Columns.id] = self.df.index

    def aggregate(self):
        self.add_dummies()
        weight_col = self.test_info.get_weight_col() or Columns.weight
        id_cols = self.test_info.get_id_cols() or Columns.id
        df = self.df.copy(deep=False)
        side_col = self.test_info.side_col
        metrics = self.test_info.metrics
        df[metrics] = df[metrics].multiply(df[side_col].map({'A': -1, 'B': 1}), axis=0)

        def weighted_avg(group):
            weights = group[weight_col]
            weighted_metrics = group[metrics].multiply(weights, axis=0)
            avg = weighted_metrics.sum() / weights.sum()
            weight_sum = weights.sum()
            return pd.Series({**avg, weight_col: weight_sum})

        result = df.groupby(id_cols).apply(weighted_avg)
        result = result.reset_index()

        return result

    @staticmethod
    def weighted_percentile(values, weights, percentile):
        """Compute the weighted percentile of a list of values."""
        sorter = np.argsort(values)
        values = values[sorter]
        weights = weights[sorter]
        weighted_quantiles = np.cumsum(weights) - 0.5 * weights
        percentile_values = np.interp(percentile, weighted_quantiles / np.sum(weights), values)
        return percentile_values

    def trim(self):
        aggregated = self.aggregate()

        trimming_list = self.test_info.trimming_list  # [0, 0.01, .... ]
        weight_col = self.test_info.get_weight_col() or Columns.weight  # string
        metrics = self.get_metric_cols()

        trimmed_cols = []
        trimmed_col_by_metric = {}
        for m in metrics:
            trimmed_col_by_metric[m] = []
            for p in trimming_list:
                trimmed_col = f"{m}_trim_{100 * p:.2f}"
                q = [p / 2, 1 - p / 2]
                trimmed_vals = self.weighted_percentile(aggregated[m], aggregated[weight_col], q)
                trimmed = np.clip(aggregated[m], *trimmed_vals)

                if len(np.unique(trimmed)) == 1:
                    pass
                else:
                    aggregated[trimmed_col] = trimmed
                    trimmed_cols.append(trimmed_col)
                    trimmed_col_by_metric[m].extend([trimmed_col])

        return aggregated, trimmed_cols, trimmed_col_by_metric

    def get_weight_col(self):
        return self.test_info.get_weight_col()

    def validate_cols(self):
        all_cols = self.test_info.get_all_cols()
        df_cols = list(self.df.columns)
        # Check if all columns in all_cols exist in df_cols
        diff_cols = set(all_cols) - set(df_cols)
        if diff_cols:
            # Raise an error with the list of columns not in df
            error_msg = f"The following columns are not in the provided dataframe: {list(diff_cols)}"
            raise ValueError(error_msg)
        else:
            # All columns exist in df, so pass
            pass

    def cast_cols(self):
        metrics = self.get_metric_cols()
        df = self.df
        df[metrics] = df[metrics].astype(float)
        self.df = df

    def get_lead_metric(self):
        return self.test_info.metrics[0]

    def get_metric_cols(self):
        return self.test_info.metrics

    def run(self):
        self.validate_cols()
        self.cast_cols()
