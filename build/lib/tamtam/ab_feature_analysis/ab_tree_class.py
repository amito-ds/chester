from tamtam.ab_info.ab_class import ABInfo
from tamtam.user_class.user_class import TestInfo


class ABTreePlot:
    def __init__(self, ab_info: ABInfo, test_info: TestInfo, metric):
        self.ab_info = ab_info
        self.test_info = test_info
        self.metric = metric
        self.df = self.ab_info.df
        self.feature_cols = self.test_info.feature_cols


    def run(self):
        pass
