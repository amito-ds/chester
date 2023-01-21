import warnings

import nltk

from features_engineering.feature_main import FeatureExtraction
from full_cycle.chapter_messages import chapter_message
from text_analyzer.data_quality import TextAnalyzer

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
import logging

logging.basicConfig(level=logging.WARNING)

from text_analyzer.smart_text_analyzer import analyze_text_df
from data_loader.webtext_data import load_data_pirates, load_data_king_arthur
from mdoel_training.best_model import ModelCycle
from mdoel_training.data_preparation import CVData
from mdoel_training.model_utils import analyze_results
from model_analyzer.model_analysis import analyze_model
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from cleaning import cleaning_func as cln
from preprocessing import preprocessing_func as pp
from features_engineering import feature_main as fe_main
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


def run_tcap(
        data_spec: DataSpec = None,
        text_cleaner: cln.TextCleaner = None, is_text_cleaner: bool = True,
        text_preprocesser: pp.TextPreprocessor = None, is_text_preprocesser: bool = True,
        text_analyzer: TextAnalyzer = None, is_text_stats: bool = True,
        feature_extraction: fe_main.FeatureExtraction = None, is_feature_extraction: bool = True,
        feature_analysis: feature_correlation.PreModelAnalysis = None, is_feature_analysis: bool = True,
        cv_data: data_preparation.CVData = None,
        model_cycle: best_model.ModelCycle = None, is_train_model: bool = True,
        is_model_analysis: bool = True
):
    """
    This function runs the full data processing and model training pipeline. It takes a series of optional inputs
    (text_cleaner, text_preprocesser, text_analyzer, feature_extraction, feature_analysis, cv_data, model_cycle)
    and applies them to the data according to the corresponding flags (is_text_cleaner, is_text_preprocesser,
    is_text_stats, is_feature_extraction, is_feature_analysis, is_train_model, is_model_analysis). The function also
    takes an optional input 'data_spec' which can be used to pass any additional information required for the pipeline.
    The function returns the cleaned and preprocessed dataframe after all the processing steps have been applied.

    Parameters:
    data_spec (DataSpec, optional): An object containing the data, text column and (optional) target column
    text_cleaner (cleaning.cleaning.TextCleaner, optional): An object containing the text cleaning settings.
    is_text_cleaner (bool, optional): A flag indicating whether text cleaning should be applied.
    text_preprocesser (preprocessing.preprocessing.TextPreprocessor, optional): An object containing the text preprocessing settings.
    is_text_preprocesser (bool, optional): A flag indicating whether text preprocessing should be applied.
    text_analyzer (TextAnalyzer, optional): An object containing the text analysis settings.
    is_text_stats (bool, optional): A flag indicating whether text statistics should be generated.
    feature_extraction (features_engineering.fe_main.FeatureExtraction, optional): An object containing the feature extraction settings.
    is_feature_extraction (bool, optional): A flag indicating whether feature extraction should be applied.
    feature_analysis (feature_analyzing.feature_correlation.PreModelAnalysis, optional): An object containing the feature analysis settings.
    is_feature_analysis (bool, optional): A flag indicating whether feature analysis should be applied.
    cv_data (mdoel_training.data_preparation.CVData, optional): An object containing the data for cross-validation.
    model_cycle (model_training.best_model.ModelCycle, optional): An object containing the model training settings.
    is_train_model (bool, optional): A flag indicating whether model training should be applied.
    is_model_analysis (bool, optional): A flag indicating whether model analysis should be applied.

    Actions:
    perform requested tasks

    """
    # Step 0: prepare outputs
    df, train_embedding, test_embedding = None, None, None

    # Step 1: Prepare text_cleaner object
    if not text_cleaner:
        text_cleaner = cln.TextCleaner()
    if data_spec:
        parameter_completer(data_spec, text_cleaner)
    origin_df = text_cleaner.df.copy()

    # Step 2: Apply Text Cleaning
    if is_text_cleaner:
        print(chapter_message("cleaning"))
        text_cleaner.generate_report()
        df = cln.clean_text_df(text_cleaner)

    # Step 3: Prepare text_preprocesser object
    if is_text_preprocesser:
        if not text_preprocesser:
            text_preprocesser = pp.TextPreprocessor()
        parameter_completer(text_cleaner, text_preprocesser)
        # pp
        print(chapter_message("preprocessing"))
        text_preprocesser.generate_report()
        df = pp.preprocess_text_df(text_preprocesser)

    # Step 4: Prepare text_analyzer object
    if is_text_stats:
        if not text_analyzer:
            text_analyzer = TextAnalyzer()
        parameter_completer(text_cleaner, text_analyzer)
        parameter_completer(text_preprocesser, text_analyzer)
        print(chapter_message("text analyze"))
        analyze_text_df(text_analyzer)

    # Step 5: Feature extraction
    if is_feature_extraction:
        if not feature_extraction:
            feature_extraction = fe_main.FeatureExtraction(training_data=df)
        if feature_extraction.training_data is None:
            feature_extraction.training_data = df
        parameter_super_completer([text_cleaner, text_preprocesser], feature_extraction)
        print(chapter_message("create embedding"))
        train_embedding, test_embedding = fe_main.extract_features(feature_extraction)

    # Step 6: Feature analysis and preparation for model training
    if is_feature_analysis or is_model_analysis or is_train_model:
        target_column = feature_extraction.target_column or 'target'
        label_encoder = LabelEncoder()
        train_embedding[target_column] = label_encoder.fit_transform(train_embedding[target_column])
        test_embedding[target_column] = label_encoder.transform(test_embedding[target_column])

    # Pre model analysis
    if is_feature_analysis:
        if not feature_analysis:
            feature_analysis = feature_correlation.PreModelAnalysis(df=train_embedding,
                                                                    target_column=target_column)
        print(chapter_message("model pre analysis"))
        feature_analysis.generate_report()
        feature_analysis.run()

    # Step 7: Model training
    best_model = None
    if is_train_model:
        print(chapter_message("model run"))
        if not cv_data:
            cv_data = CVData(train_data=train_embedding, test_data=test_embedding)
        if not model_cycle:
            model_cycle = ModelCycle(cv_data=cv_data, target_col=target_column)
        best_model = model_cycle.get_best_model()
        print("Winning model: ", best_model.model_name)

    # Step 8: Model analysis
    if is_model_analysis:
        print(chapter_message("post model analysis"))
        organized_results = pd.DataFrame(best_model.results)
        analyze_results(organized_results, best_model.parameters)
        analyze_model(best_model.model, cv_data, target_label=target_column)
    return origin_df, df, train_embedding, test_embedding, best_model


# # #
# df1 = load_data_pirates().assign(target='chat_logs')
# df2 = load_data_king_arthur().assign(target='pirates')
# df = pd.concat([df1, df2])

import pandas as pd
# nltk.download('brown')
from nltk.corpus import brown

# Create an empty list to store the data
data = []

# Iterate through the samples and add the text and category to the data list
# Select the categories to sample from
categories = ['news', 'romance', 'science_fiction']
for category in categories:
    for text in brown.sents(categories=category)[:1000]:
        data.append({'text': ' '.join(text), 'target': category})

# Create a dataframe from the data list
df = pd.DataFrame(data)

# Print the first 5 rows of the dataframe
print(df.head())

out = run_tcap(
    data_spec=DataSpec(df=df, text_column='text', target_column='target'),
)

# out = run_tcap(
#     data_spec=DataSpec(df=df, text_column='text', target_column='target'),
#     feature_extraction=FeatureExtraction(split_data=False),
#     is_model_analysis=False, is_train_model=False, is_feature_analysis=False
# )

# concat
# embedding_with_label = pd.concat([origin_df.reset_index(), train_embedding.reset_index()], axis = 0)

