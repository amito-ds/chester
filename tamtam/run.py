from tamtam.ab_info.ab_class import ABInfo
from tamtam.allocation_calculation.bias_class import BiasCalculation
from tamtam.delta_analysis.delta_class import DeltaPValueCalc
from tamtam.metrics_correlation.metric_class import MetricCorrelation
from tamtam.user_class.user_class import ABData, TestInfo


def run(ab_data: ABData, test_info: TestInfo):
    # load data
    # check if columns in pandas
    # casting to dates, numerics, more
    # classify the features
    # put it all in AB test info

    # AB info
    ab_info = ABInfo(ab_data=ab_data, test_info=test_info)

    # calculate bias
    # BiasCalculation(ab_info).run()

    # metrics corr
    # MetricCorrelation(ab_info).run()

    # Delta, Pvalue
    DeltaPValueCalc(ab_info=ab_info, test_info=test_info).run()

