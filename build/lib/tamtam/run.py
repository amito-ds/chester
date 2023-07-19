from tamtam.ab_feature_analysis.ab_feature_analysis_class import ABFeatureAnalysis
from tamtam.ab_info.ab_class import ABInfo
from tamtam.allocation_calculation.bias_class import BiasCalculation
from tamtam.delta_analysis.delta_class import DeltaPValueCalc
from tamtam.metrics_correlation.metric_class import MetricCorrelation
from tamtam.user_class.user_class import ABData, TestInfo


def run(ab_data: ABData, test_info: TestInfo):
    # AB info
    ab_info = ABInfo(ab_data=ab_data, test_info=test_info)

    # calculate bias
    print("==========================> Bias calculations: ")
    BiasCalculation(ab_info).run()

    # metrics corr
    MetricCorrelation(ab_info).run()

    # Delta, Pvalue
    print("==========================>  Test Analysis ")
    DeltaPValueCalc(ab_info=ab_info, test_info=test_info).run()

    # Features
    if len(ab_info.test_info.get_feature_cols()) > 0:
        ABFeatureAnalysis(ab_info=ab_info, test_info=test_info).run()
