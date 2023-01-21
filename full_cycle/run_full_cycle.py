import warnings

from text_analyzer.data_quality import TextAnalyzer

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
import logging

# logging.basicConfig(filename='lightgbm.log', level=logging.WARNING)
# logging.basicConfig(level=logging.WARNING)
logging.basicConfig(level=logging.ERROR)
from text_analyzer.smart_text_analyzer import analyze_text, analyze_text_df
from data_loader.webtext_data import load_data_pirates, load_data_king_arthur
from feature_analyzing.feature_correlation import PreModelAnalysis
from features_engineering.fe_main import extract_features, FeatureExtraction
from mdoel_training.best_model import ModelCycle
from mdoel_training.data_preparation import CVData
from mdoel_training.model_utils import analyze_results
from model_analyzer.model_analysis import analyze_model
from preprocessing.preprocessing import preprocess_text_df, TextPreprocessor
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from cleaning.cleaning import clean_text_df, TextCleaner
from cleaning import cleaning as cln
from preprocessing import preprocessing as pp
from features_engineering import fe_main
from feature_analyzing import feature_correlation
from mdoel_training import data_preparation
from mdoel_training import best_model


def parameter_completer(instance1, instance2):
    attributes1 = vars(instance1)
    attributes2 = vars(instance2)
    for key in attributes1:
        if key in attributes2:
            setattr(instance2, key, getattr(instance1, key))


def parameter_super_completer(instance_list: list, instance):
    for instance_i in instance_list:
        parameter_completer(instance_i, instance)


class DataSpec:
    def __init__(self, df: pd.DataFrame, text_column: str = 'text', target_column: str = None):
        self.df = df
        self.text_column = text_column
        self.target_column = target_column


def run_full_cycle(
        data_spec: DataSpec = None,
        text_cleaner: cln.TextCleaner = None, is_text_cleaner: bool = True,
        text_preprocesser: pp.TextPreprocessor = None, is_text_preprocesser: bool = True,
        text_analyzer: TextAnalyzer = None, is_text_stats: bool = True,
        feature_extraction: fe_main.FeatureExtraction = None, is_feature_extraction: bool = False,
        feature_analysis: feature_correlation.PreModelAnalysis = None, is_feature_analysis: bool = False,
        cv_data: data_preparation.CVData = None,
        model_cycle: best_model.ModelCycle = None, is_train_model: bool = False,
        is_model_analysis: bool = False
):
    if not data_spec:
        parameter_completer(data_spec, text_cleaner)
    ###### clean ######
    if not text_cleaner:
        text_cleaner = TextCleaner()
    if data_spec:
        parameter_completer(data_spec, text_cleaner)
    if is_text_cleaner:
        df = clean_text_df(text_cleaner)  # clean

    ###### preprocess ######
    if is_text_preprocesser:
        if not text_preprocesser:
            text_preprocesser = TextPreprocessor()
        parameter_completer(text_cleaner, text_preprocesser)
        # pp
        df = preprocess_text_df(text_preprocesser)

    ## Text stats
    if not text_analyzer:
        text_analyzer = TextAnalyzer()
    parameter_completer(text_cleaner, text_analyzer)
    parameter_completer(text_preprocesser, text_analyzer)
    if is_text_stats:
        analyze_text_df(text_analyzer)

    ##### Feature extraction
    if is_feature_extraction:
        if not feature_extraction:
            feature_extraction = FeatureExtraction(training_data=df)
        parameter_super_completer([text_cleaner, text_preprocesser], feature_extraction)
        train_embedding, test_embedding = extract_features(feature_extraction)

    if is_feature_analysis or is_model_analysis or is_train_model:
        target_column = feature_extraction.target_column or 'target'
        label_encoder = LabelEncoder()
        train_embedding[target_column] = label_encoder.fit_transform(train_embedding[target_column])
        test_embedding[target_column] = label_encoder.transform(test_embedding[target_column])

    # Pre model analysis
    if is_feature_analysis:
        if not feature_analysis:
            feature_analysis = PreModelAnalysis(df=train_embedding,
                                                target_column=target_column)
        feature_analysis.run()

    # training a model
    best_model = None
    if is_train_model:
        if not cv_data:
            cv_data = CVData(train_data=train_embedding, test_data=test_embedding)
        if not model_cycle:
            model_cycle = ModelCycle(cv_data=cv_data, target_col=target_column)
        best_model = model_cycle.get_best_model()
        print("best_model", best_model.model_name)

    # step 6: analyze the results and the best model
    if is_model_analysis:
        organized_results = pd.DataFrame(best_model.results)
        analyze_results(organized_results, best_model.parameters)
        analyze_model(best_model.model, cv_data, target_label=target_column)

    return df


# # #
df1 = load_data_pirates().assign(target='chat_logs')
df2 = load_data_king_arthur().assign(target='pirates')
df = pd.concat([df1, df2])

run_full_cycle(DataSpec(df=df, text_column='text', target_column='target'))
