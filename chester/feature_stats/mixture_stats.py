from chester.zero_break.problem_specification import DataInfo


class NumericStats:
    def __init__(self, data_info: DataInfo):
        self.data_info = data_info
        self.cols = self.data_info.feature_types_val["numeric"]
        self.data = self.data_info.data[self.cols]
        # self.cols_sorted = self.sort_by_variance()
        # sort numerical
        # sort categorical
        # when sample n:
        # if total cat + numeric < n => take cat+numerics
        # if both > n, take first n from each and then sample n
        # if one is small and the other is large, take all the small one and sample the rest of double from the other
        # corr: candle, sort numerics and then categorical
