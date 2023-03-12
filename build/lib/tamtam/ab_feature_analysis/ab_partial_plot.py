from tamtam.ab_info.ab_class import ABInfo
from tamtam.user_class.user_class import TestInfo


class ABPartialPlot:
    def __init__(self, ab_info: ABInfo, test_info: TestInfo, metric):
        self.ab_info = ab_info
        self.test_info = test_info
        self.metric = metric
        self.df = self.ab_info.df

    def run(self):
        pass
