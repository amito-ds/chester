import warnings

import IPython
from matplotlib.backends.backend_pdf import PdfPages


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

from cleaning import cleaning as cln
from preprocessing import preprocessing as pp
from features_engineering import fe_main
from feature_analyzing import feature_correlation
from mdoel_training import data_preparation
from mdoel_training import best_model


# mute warnings
# warnings.filterwarnings("ignore", category=UserWarning, module='lightgbm')


def parameter_completer(instance1, instance2):
    attributes1 = vars(instance1)
    attributes2 = vars(instance2)
    for key in attributes1:
        if key in attributes2:
            setattr(instance2, key, getattr(instance1, key))
            print(f'attribute {key} updated on instance2')


def parameter_super_completer(instance, instance_list: list):
    for instance_i in instance_list:
        parameter_completer(instance_i, instance)


def run_full_cycle(text_cleaner: cln.TextCleaner,
                   text_preprocesser: pp.TextPreprocessor = None,
                   feature_extraction: fe_main.FeatureExtraction = None,
                   feature_analysis: feature_correlation.PreModelAnalysis = None,
                   cv_data: data_preparation.CVData = None,
                   model_cycle: best_model.ModelCycle = None,
                   ):
    ###### clean ######
    df = clean_text_df(text_cleaner)  # clean

    ###### preprocess ######
    if not text_preprocesser:
        text_preprocesser = TextPreprocessor()
    parameter_super_completer(text_preprocesser, [text_cleaner])
    # pp
    df = preprocess_text_df(text_preprocesser)
    #
    # #### Feature extraction
    if not feature_extraction:
        feature_extraction = FeatureExtraction(training_data=df)
    parameter_super_completer(feature_extraction, [text_cleaner])
    train_embedding, test_embedding = extract_features(feature_extraction)

    # print("train_embedding shape", train_embedding.shape)
    # print("show ", train_embedding[0:10])
    #
    # ### Pre model analysis
    target_column = feature_extraction.target_column
    label_encoder = LabelEncoder()
    train_embedding[target_column] = label_encoder.fit_transform(train_embedding[target_column])
    test_embedding[target_column] = label_encoder.transform(test_embedding[target_column])
    if not feature_analysis:
        feature_analysis = PreModelAnalysis(df=train_embedding,
                                            target_column=target_column)
    # feature_analysis.run()
    #
    # ### training a model
    if not cv_data:
        cv_data = CVData(train_data=train_embedding, test_data=test_embedding)
    if not model_cycle:
        model_cycle = ModelCycle(cv_data=cv_data, target_col=target_column)
    best_model = model_cycle.get_best_model()
    print("best_model", best_model.model_name)

    # step 6: analyze the results and the best model
    organized_results = pd.DataFrame(best_model.results)
    analyze_results(organized_results, best_model.parameters)
    analyze_model(best_model.model, cv_data, target_label='target')

    return df


import pandas as pd
from sklearn.preprocessing import LabelEncoder

from cleaning.cleaning import clean_text, clean_text_df, TextCleaner
from data_loader.webtext_data import load_data_pirates, load_data_king_arthur, load_data_chat_logs
from feature_analyzing.feature_correlation import PreModelAnalysis
from features_engineering.fe_main import get_embeddings, extract_features, FeatureExtraction
from mdoel_training.best_model import ModelCycle
from mdoel_training.data_preparation import CVData
from mdoel_training.model_utils import analyze_results
from model_analyzer.model_analysis import analyze_model
from preprocessing.preprocessing import preprocess_text, get_stemmer, preprocess_text_df, TextPreprocessor
from util import get_stopwords

#
# #
# # #
df1 = load_data_chat_logs().assign(target='chat_logs').sample(500, replace=True)
df2 = load_data_king_arthur().assign(target='wow').sample(100, replace=True)
df3 = load_data_pirates().assign(target='pirate').sample(500, replace=True)
df4 = load_data_chat_logs().assign(target='chat_logs_b').sample(100, replace=True)

df = pd.concat([
    df1,
    df2,
    # df3,
    # df4
])

import sys

# Save the original stdout
original_stdout = sys.stdout

# Open a file for writing
import logging

# Configure the logging module
logging.basicConfig(filename='output.log', level=logging.WARNING)
logging.basicConfig(filename='output.log', level=logging.WARN)

run_full_cycle(TextCleaner(df=df))

