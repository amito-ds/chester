from chester.text_stats_analysis.data_quality import TextAnalyzer
from chester.text_stats_analysis.smart_text_analyzer import analyze_text_df
from chester.zero_break.problem_specification import DataInfo


class TextStats:
    def __init__(self, data_info: DataInfo):
        self.data_info = data_info
        self.cols = self.data_info.feature_types_val["text"]
        self.data = self.data_info.data  # [self.cols]

    def run(self):
        for col in self.cols:
            print(f"** Analayzing {col} column")
            text_analyzer = TextAnalyzer(df=self.data, text_column=col)
            analyze_text_df(text_analyzer)
