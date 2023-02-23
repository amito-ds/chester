from tamtam.utils.utils import true_if_exists


class ABData:
    def __init__(self, df):
        self.df = df


class TestInfo:
    def __init__(self, side_col, metrics: list,
                 id_cols=None, feature_cols=None, weight_col=None, date_col=None, is_higher_better=True,
                 trimming_list=None):
        self.side_col = side_col
        self.metrics = metrics
        self.id_cols = id_cols
        self.feature_cols = feature_cols
        self.weight_col = weight_col
        self.date_col = date_col
        self.is_higher_better = is_higher_better
        self.trimming_list = trimming_list
        if not self.trimming_list:
            self.trimming_list = [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

    def get_side_col(self):
        if not self.side_col:
            return []
        return [self.side_col]

    def get_metrics(self):
        if not self.metrics:
            return []
        return self.metrics

    def get_id_cols(self):
        if not self.id_cols:
            return []
        return self.id_cols

    def is_id_cols(self):
        return true_if_exists(self.id_cols)

    def get_feature_cols(self):
        if not self.feature_cols:
            return []
        return self.feature_cols

    def get_weight_col(self):
        return self.weight_col

    def is_weight_col(self):
        return true_if_exists(self.weight_col)

    def get_date_col(self):
        if not self.date_col:
            return []
        return self.date_col

    def get_is_higher_better(self):
        return self.is_higher_better

    def get_all_cols(self):
        all_cols = self.get_side_col() + self.get_metrics() + self.get_id_cols() + self.get_feature_cols()
        if self.get_weight_col():
            all_cols.append(self.get_weight_col())
        if self.get_date_col():
            all_cols.append(self.get_date_col())
        return all_cols
