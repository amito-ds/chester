from chester.run.user_classes import TextFeatureExtraction
from chester.text_stats_analysis.data_quality import TextAnalyzer
from chester.text_stats_analysis.smart_text_analyzer import analyze_text_df
from chester.util import ReportCollector, REPORT_PATH
from chester.zero_break.problem_specification import DataInfo


class TextStats:
    def __init__(self, data_info: DataInfo, text_spec: TextFeatureExtraction = None):
        self.data_info = data_info
        self.text_spec = text_spec
        self.cols = self.data_info.feature_types_val["text"]
        self.data = self.data_info.data

    def run(self):
        rc = ReportCollector(REPORT_PATH)
        rc.save_text("Text Statistics:")
        rc.save_text("Extracting Embedding")
        for col in self.cols:
            title_to_print = f"** Analyzing {col} Column"
            print(title_to_print)
            rc.save_text(title_to_print)
            if self.text_spec is None:
                text_analyzer = TextAnalyzer(df=self.data, text_column=col)
            else:
                corex_topics_num = self.text_spec.corex_dim
                corex_anchor_words = self.text_spec.anchor_words
                corex_anchor_strength = self.text_spec.anchor_strength
                text_analyzer = TextAnalyzer(df=self.data, text_column=col, corex_topics=corex_topics_num,
                                             corex_topics_num=corex_topics_num,
                                             corex_anchor_strength=corex_anchor_strength,
                                             corex_anchor_words=corex_anchor_words)

            analyze_text_df(text_analyzer)
