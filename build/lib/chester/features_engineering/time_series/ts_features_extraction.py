from chester.features_engineering.time_series.ts_feature_extraction import TimeSeriesFeatureExtraction
from chester.run.user_classes import TimeSeriesHandler
from chester.zero_break.problem_specification import DataInfo


class TimeSeriesFeaturesExtraction:
    def __init__(self,
                 time_series_handler: TimeSeriesHandler = None,
                 data_info: DataInfo = None
                 ):
        self.time_series_handler = time_series_handler or TimeSeriesHandler()
        self.data_info = data_info
        self.cols = self.data_info.feature_types_val["time"]

    def run(self):
        for col in self.cols:
            ts_fe = TimeSeriesFeatureExtraction(
                data_info=self.data_info,
                time_series_handler=self.time_series_handler,
                col_name=col,
                column=self.data_info.data[col]
            )
            ts_fe.run()
            self.data_info = ts_fe.data_info
            # print("updated data info after ts", self.data_info)
