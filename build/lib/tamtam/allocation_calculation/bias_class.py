import pandas as pd

from tamtam.ab_info.ab_class import ABInfo


class BiasCalculation:
    def __init__(self, ab_info: ABInfo):
        self.ab_info = ab_info
        self.df = self.ab_info.df
        self.side_col = self.ab_info.test_info.get_side_col()[0]
        self.control = "A"
        self.treatment = "B"

    def calculate_bias(self):
        side_col = self.side_col
        id_cols = self.ab_info.test_info.get_id_cols()
        weight_col = self.ab_info.test_info.get_weight_col()
        df = self.df.copy()

        # Calculate count bias
        control_df = df.loc[df[side_col] == self.control]
        treatment_df = df.loc[df[side_col] == self.treatment]

        control_count = control_df.shape[0]
        treatment_count = treatment_df.shape[0]
        count_bias = treatment_count / control_count

        # Calculate ID bias
        control_ids = control_df[id_cols].drop_duplicates().shape[0]
        treatment_ids = treatment_df[id_cols].drop_duplicates().shape[0]
        id_bias = treatment_ids / control_ids

        # Calculate weight bias, if weight column exists
        if weight_col:
            control_weight = control_df[weight_col].sum()
            treatment_weight = treatment_df[weight_col].sum()
            weight_bias = treatment_weight / control_weight
        else:
            control_weight = control_count
            treatment_weight = treatment_count
            weight_bias = id_bias

        # Create a pandas dataframe with the bias values
        bias_df = pd.DataFrame({
            'bias_type': ['count', 'ids', 'weights'],
            'A': [control_count, control_ids, control_weight],
            'B': [treatment_count, treatment_ids, treatment_weight],
            'B/A - 1': [count_bias - 1, id_bias - 1, weight_bias - 1 if weight_bias else None]
        })

        return bias_df

    def run(self):
        print("Bias calculations")
        print(self.calculate_bias())
