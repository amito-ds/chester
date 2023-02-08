import warnings

from chester.cleaning.cleaner_handler import CleanerHandler
from chester.feature_stats.categorical_stats import CategoricalStats
from chester.feature_stats.numeric_stats import NumericStats
from chester.feature_stats.text_stats import TextStats
from chester.features_engineering.features_handler import FeaturesHandler
from chester.model_monitor.mm_bootstrap import ModelBootstrap
from chester.model_monitor.mm_weaknesses import ModelWeaknesses
from chester.model_training.models.chester_models.best_model import BestModel
from chester.post_model_analysis.post_model_analysis_class import PostModelAnalysis
from chester.pre_model_analysis.categorical import CategoricalPreModelAnalysis
from chester.pre_model_analysis.numerics import NumericPreModelAnalysis
from chester.pre_model_analysis.target import TargetPreModelAnalysis
from chester.preprocessing.preprocessor_handler import PreprocessHandler
from chester.run.user_classes import Data, TextHandler, TextAnalyze, FeatureStats, PreModel, ModelRun, PostModel, \
    ModelWeakness
from chester.zero_break.problem_specification import DataInfo

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
import logging

logging.basicConfig(level=logging.WARNING)

from chester.model_training.data_preparation import CVData
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def run_chester(
        data_spec: Data = None,
        feature_types: dict = None,
        text_handler: TextHandler = None,
        text_analyze: TextAnalyze = None,
        feature_stats: FeatureStats = None,
        pre_model: PreModel = None,
        model_run: ModelRun = None,
        post_model: PostModel = None,
        model_weaknesses: ModelWeakness = None,
        # special params
        plot=None,
        max_stats_show = None,
        # text_cleaner: cln.TextCleaner = None, is_text_cleaner: bool = True,
        # text_preprocesser: pp.TextPreprocessor = None, is_text_preprocesser: bool = True,
        # text_analyzer: TextAnalyzer = None, is_text_stats: bool = True,
        # feature_extraction: fe_main.FeatureExtraction = None, is_feature_extraction: bool = True,
        # feature_analysis: feature_correlation.PreModelAnalysis = None, is_feature_analysis: bool = True,
        # cv_data: data_preparation.CVData = None,
        # model_cycle: bm.ModelCycle = None, model_compare: bm.CompareModels = None,
        # is_train_model: bool = True,
        # is_model_analysis: bool = True
):
    # chapters: ZB, Text stats, Feature stats, pre model, model, post model, model weaknesses (7 chapters)
    # subchapters: ZB,

    #################################################### meta learn
    # TO DO: print meta learn message + shuffle message
    df = data_spec.df.sample(frac=1).reset_index(drop=True)
    data_info = DataInfo(data=df, target='target')
    data_info.calculate()

    if feature_types is not None:
        data_info.feature_types_val = feature_types
    target_column = data_info.target
    print(data_info)
    ####################################################

    #################################################### Text handling
    # TO DO: print message
    ## cleaning
    cleaner = CleanerHandler(data_info)
    cleaner.transform()
    data_info = cleaner.data_info

    ## keep a copy of all text after cleaning
    text_cols = data_info.feature_types_val["text"]
    clean_text_df = pd.DataFrame()
    if len(text_cols) > 0:
        pd.options.mode.chained_assignment = None
        clean_text_df = data_info.data[text_cols]
        clean_text_df.rename(columns={col: "clean_" + col for col in clean_text_df.columns}, inplace=True)

    ## preprocessing
    pp = PreprocessHandler(data_info)
    pp.transform()
    data_info_original = data_info
    data_info = pp.data_info

    # data_info_text_cleaning for later text analysis
    if len(text_cols) > 0:
        clean_text_df = pd.concat([df, clean_text_df], axis=1)
    ####################################################

    #################################################### Feat extract
    # TO DO: print message
    feat_hand = FeaturesHandler(data_info)
    feature_types, final_df = feat_hand.transform()
    final_df[target_column] = data_info.data[data_info.target]
    ####################################################

    #################################################### Feat Stats
    # TO DO: print message
    data_info_num_stats = DataInfo(data=final_df, target=target_column)
    data_info_num_stats.calculate()

    num_cols = data_info_num_stats.feature_types_val["numeric"]
    cat_cols = data_info.feature_types_val["categorical"]
    text_cols = data_info.feature_types_val["text"]

    if len(num_cols) > 0:
        print("Numerical Feature statistics")
        NumericStats(data_info_num_stats).run()
    print("Categorical Feature statistics")
    CategoricalStats(data_info).run()
    if len(text_cols) > 0:
        print("Text Feature statistics")
        data_info.data = clean_text_df
        TextStats(data_info).run()
    ####################################################

    #################################################### Pre model
    # TO DO: print message
    ## label stats
    TargetPreModelAnalysis(data_info).run()
    ## num, cat pre model
    NumericPreModelAnalysis(data_info_num_stats).run()
    data_info.data = df
    CategoricalPreModelAnalysis(data_info).run()
    ####################################################

    #################################################### model
    # TO DO: print message

    # encode if needed
    if data_info.problem_type_val in ["Binary classification", "Multiclass classification"]:
        print("Encoding target")
        label_encoder = LabelEncoder()
        final_df[target_column] = label_encoder.fit_transform(final_df[target_column])

    # cv data
    cv_data = CVData(train_data=final_df, test_data=None, target_column='target', split_data=True)
    data_info.feature_types_val = feature_types
    # run the model
    model = BestModel(data_info=data_info, cv_data=cv_data, num_models_to_compare=3)
    model_results = model.get_best_model()  # returns resultf of the best baseline model
    # print model metadata
    params = model_results[1].get_params()
    print(f"Best model: {type(model_results[1])}, with parameters:")
    for p in params:
        print(p.name, ":", p.value)
    ####################################################

    #################################################### post model
    # TO DO: print message
    PostModelAnalysis(cv_data, data_info, model=model_results[1]).analyze()
    ModelBootstrap(cv_data, data_info, model=model_results[1]).plot()
    ####################################################

    #################################################### model weaknesses
    # TO DO: print message
    model_weaknesses = ModelWeaknesses(cv_data, data_info, model=model_results[1])
    model_weaknesses.run()
    ####################################################
