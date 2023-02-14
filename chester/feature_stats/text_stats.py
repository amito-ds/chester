from chester.text_stats_analysis.data_quality import TextAnalyzer
from chester.text_stats_analysis.smart_text_analyzer import analyze_text_df
from chester.util import ReportCollector, REPORT_PATH
from chester.zero_break.problem_specification import DataInfo


class TextStats:
    def __init__(self, data_info: DataInfo):
        self.data_info = data_info
        self.cols = self.data_info.feature_types_val["text"]
        self.data = self.data_info.data  # [self.cols]

    def run(self):
        rc = ReportCollector(REPORT_PATH)
        rc.save_text("Text statistics:")
        rc.save_text("Extracting embedding")
        for col in self.cols:
            title_to_print = f"** Analayzing {col} column"
            print(title_to_print)
            rc.save_text(title_to_print)

            text_analyzer = TextAnalyzer(df=self.data, text_column=col)
            analyze_text_df(text_analyzer)
