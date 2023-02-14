from chester.preprocessing import preprocessing_func as pp
from chester.util import ReportCollector, REPORT_PATH
from chester.zero_break.problem_specification import DataInfo


class PreprocessHandler:
    def __init__(self, data_info: DataInfo, text_pre_process: pp.TextPreprocessor = None):
        self.data_info = data_info
        self.text_pre_process = text_pre_process
        if self.text_pre_process is not None:
            self.text_pre_process.df = self.data_info.data
        self.target = data_info.target
        self.problem_type_val = data_info.problem_type_val
        self.feature_types_val = data_info.feature_types_val
        self.loss_detector_val = data_info.loss_detector_val
        self.metrics_detector_val = data_info.metrics_detector_val
        self.model_selection_val = data_info.model_selection_val
        self.label_transformation_val = data_info.label_transformation_val

    def transform(self):
        text_columns = self.feature_types_val.get("text")
        for col in text_columns:
            if self.text_pre_process is not None:
                curr_text_pre_processor = self.text_pre_process
                curr_text_pre_processor.text_column = col
                text_reprocess = curr_text_pre_processor
            else:
                text_reprocess = pp.TextPreprocessor(self.data_info.data, text_column=col)
            title_to_print = f"{col} column preprocessing"
            print(title_to_print)
            rc = ReportCollector(REPORT_PATH)
            rc.save_text(title_to_print)
            text_reprocess.generate_report()
            self.data_info.data = pp.preprocess_text_df(text_reprocess)
