from chester.zero_break.problem_specification import DataInfo


class CategoricalStats:
    def __init__(self, data_info: DataInfo):
        self.data_info = data_info
        self.cols = list(
            set(self.data_info.feature_types_val["categorical"] + self.data_info.feature_types_val["boolean"]))
        self.data = self.data_info.data[self.cols]

    # to calculate per feature:
    # number of unique values
    # of missing
    # Top 3 common values + text limitation
    # how many top X values covers?
    # plots:
    # matrix corr spearman
    # for top 9 features with the highest var: plot 3X3 histogram

