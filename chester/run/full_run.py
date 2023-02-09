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
from chester.run.user_classes import Data, TextHandler, FeatureStats, ModelRun, TextFeatureSpec, FeatureTypes
from chester.zero_break.problem_specification import DataInfo

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
import logging

logging.basicConfig(level=logging.WARNING)

from chester.model_training.data_preparation import CVData
import pandas as pd
from sklearn.preprocessing import LabelEncoder

madcat_collector = {}


def run_madcat(
        data_spec: Data = None,
        feature_types: dict = None,
        text_handler: TextHandler = None, is_text_handler=True,
        feature_stats: FeatureStats = None, is_feature_stats=True,
        text_feature_extraction: TextFeatureSpec = None,
        is_pre_model=True,
        model_run: ModelRun = None,
        is_post_model=True,
        is_model_weaknesses=True,
        # special params
        plot=None,
        max_stats_show=None,
        # model_cycle: bm.ModelCycle = None, model_compare: bm.CompareModels = None,
        # is_train_model: bool = True,
        # is_model_analysis: bool = True
):
    # chapters: ZB, Text stats, Feature stats, pre model, model, post model, model weaknesses (7 chapters)

    #################################################### meta learn
    # TO DO: print meta learn message + shuffle message
    df = data_spec.df.sample(frac=1).reset_index(drop=True)
    data_info = DataInfo(data=df, target='target')
    data_info.calculate()

    if feature_types is not None:
        data_info = FeatureTypes(feature_types).update_data_info(data_info=data_info)
    target_column = data_info.target
    print(data_info)
    madcat_collector["data info"] = data_info
    ####################################################

    #################################################### Text handling
    # TO DO: print message
    # cleaning
    if is_text_handler:
        text_cleaner = None
        text_pre_process = None
        if text_handler is not None:
            text_cleaner = text_handler.text_cleaner
            text_pre_process = text_handler.text_pre_process

        cleaner = CleanerHandler(data_info=data_info, text_cleaner=text_cleaner)
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
        pp = PreprocessHandler(data_info=data_info, text_pre_process=text_pre_process)
        pp.transform()
        data_info_original = data_info  # in case you need it
        data_info = pp.data_info

        # data_info_text_cleaning for later text analysis
        if len(text_cols) > 0:
            clean_text_df = pd.concat([df, clean_text_df], axis=1)
        madcat_collector["data info"] = data_info
    ####################################################

    #################################################### Feat extract
    # TO DO: print message
    if text_feature_extraction is not None:
        feat_hand = FeaturesHandler(data_info=data_info, text_feature_extraction=text_feature_extraction)
    else:
        feat_hand = FeaturesHandler(data_info=data_info)
    feature_types, final_df = feat_hand.transform()
    final_df[target_column] = data_info.data[data_info.target]

    madcat_collector["data info"] = data_info
    madcat_collector["features data"] = final_df
    ####################################################

    #################################################### Feat Stats
    # TO DO: print message
    data_info_num_stats = None
    if is_feature_stats:
        data_info_num_stats = DataInfo(data=final_df, target=target_column)
        data_info_num_stats.calculate()

        num_cols = data_info_num_stats.feature_types_val["numeric"]
        cat_cols = data_info.feature_types_val["categorical"]
        text_cols = data_info.feature_types_val["text"]

        # plot trick
        plot_stats = True
        if plot:
            plot_stats = True
        else:
            if feature_stats is not None:
                if feature_stats.plot is not None:
                    plot_stats = feature_stats.plot
            else:
                plot_stats = True

        if len(num_cols) > 0:
            print("Numerical Feature statistics")
            NumericStats(data_info_num_stats).run(plot=plot_stats)
        print("Categorical Feature statistics")
        CategoricalStats(data_info).run(plot=plot_stats)
        if len(text_cols) > 0:
            print("Text Feature statistics")
            data_info.data = clean_text_df
            TextStats(data_info).run()
    ####################################################

    # No target => no model!
    if data_info.problem_type_val == "No target variable":
        return madcat_collector

    #################################################### Pre model
    # TO DO: print message
    if is_pre_model:
        # label stats
        TargetPreModelAnalysis(data_info).run(plot)
        # num, cat pre model
        if data_info_num_stats is None:
            data_info_num_stats = DataInfo(data=final_df, target=target_column)
            data_info_num_stats.calculate()
        NumericPreModelAnalysis(data_info_num_stats).run(plot)
        data_info.data = df
        CategoricalPreModelAnalysis(data_info).run(plot)
    ####################################################

    #################################################### model
    # TO DO: print message
    # encode if needed
    if data_info.problem_type_val in ["Binary classification", "Multiclass classification"]:
        # print("Encoding target")
        label_encoder = LabelEncoder()
        final_df[target_column] = label_encoder.fit_transform(final_df[target_column])

    # cv data
    cv_data = CVData(train_data=final_df, test_data=None, target_column='target', split_data=True)
    data_info.feature_types_val = feature_types
    # run the model
    if model_run is not None:
        model = BestModel(data_info=data_info,
                          cv_data=cv_data,
                          num_models_to_compare=model_run.n_models,
                          best_practice_prob=model_run.best_practice_prob)
    else:
        model = BestModel(data_info=data_info, cv_data=cv_data)
    model_results = model.get_best_model()  # returns result of the best baseline model
    # print model metadata
    params = model_results[1].get_params()
    print(f"Best model: {type(model_results[1])}, with parameters:")
    for p in params:
        print(p.name, ":", p.value)

    madcat_collector["data info"] = data_info
    madcat_collector["features data"] = final_df
    madcat_collector["model"] = model
    madcat_collector["parameters"] = model_results[1]
    madcat_collector["model_results"] = model_results[0]
    ####################################################

    #################################################### post model
    # TO DO: print message
    if is_post_model:
        post_model_analysis = PostModelAnalysis(cv_data, data_info, model=model_results[1])
        madcat_collector["post_model_analysis"] = post_model_analysis
        post_model_analysis.analyze()

        model_bootstrap = ModelBootstrap(cv_data, data_info, model=model_results[1])
        madcat_collector["model_bootstrap"] = model_bootstrap
        model_bootstrap.plot()
    ####################################################

    #################################################### model weaknesses
    # TO DO: print message
    if is_model_weaknesses:
        model_weaknesses = ModelWeaknesses(cv_data, data_info, model=model_results[1])
        madcat_collector["model_weaknesses"] = model_weaknesses
        model_weaknesses.run()
    ####################################################

    return madcat_collector
