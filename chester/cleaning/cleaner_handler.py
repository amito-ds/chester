from chester.cleaning import cleaning_func as cln
from chester.run.chapter_titles import chapter_title
from chester.util import ReportCollector, REPORT_PATH

from chester.zero_break.problem_specification import DataInfo


class CleanerHandler:
    def __init__(self, data_info: DataInfo, text_cleaner: cln.TextCleaner = None):
        self.data_info = data_info
        self.text_cleaner = text_cleaner
        if text_cleaner is not None:
            self.text_cleaner.df = self.data_info.data
        self.target = data_info.target
        self.problem_type_val = data_info.problem_type_val
        self.feature_types_val = data_info.feature_types_val
        self.loss_detector_val = data_info.loss_detector_val
        self.metrics_detector_val = data_info.metrics_detector_val
        self.model_selection_val = data_info.model_selection_val
        self.label_transformation_val = data_info.label_transformation_val

    def transform(self):
        text_columns = self.feature_types_val.get("text")
        if len(text_columns) > 0:
            print(chapter_title("meta learn"))
        for col in text_columns:
            if self.text_cleaner is not None:
                curr_text_cleaner = self.text_cleaner
                curr_text_cleaner.text_column = col
                text_cleaner = curr_text_cleaner
            else:
                text_cleaner = cln.TextCleaner(self.data_info.data, text_column=col)
            title_to_print = f"Cleaning and preprocessing {col} column"
            print(title_to_print)
            rc = ReportCollector(REPORT_PATH)
            rc.save_text(title_to_print)
            text_cleaner.generate_report()
            self.data_info.data = cln.clean_text_df(text_cleaner)
