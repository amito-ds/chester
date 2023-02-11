import warnings

from chester.cleaning.cleaner_handler import CleanerHandler
from chester.feature_stats.categorical_stats import CategoricalStats
from chester.feature_stats.numeric_stats import NumericStats
from chester.feature_stats.text_stats import TextStats
from chester.features_engineering.features_handler import FeaturesHandler
from chester.model_monitor.error_prediction import ModelWeaknesses
from chester.model_monitor.model_boostrap import ModelBootstrap
from chester.model_training.models.chester_models.best_model import BestModel
from chester.post_model_analysis.post_model_analysis_class import PostModelAnalysis
from chester.pre_model_analysis.categorical import CategoricalPreModelAnalysis
from chester.pre_model_analysis.numerics import NumericPreModelAnalysis
from chester.pre_model_analysis.target import TargetPreModelAnalysis
from chester.preprocessing.preprocessor_handler import PreprocessHandler
from chester.run.chapter_titles import chapter_title
from chester.run.user_classes import Data, TextHandler, FeatureStats, ModelRun, TextFeatureSpec, FeatureTypes
from chester.zero_break.problem_specification import DataInfo

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
import logging

logging.basicConfig(level=logging.WARNING)

from chester.model_training.data_preparation import CVData
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def run_madcat(
        data_spec: Data,
        feature_types: dict = None,
        text_handler: TextHandler = None, is_text_handler=True,
        feature_stats: FeatureStats = None, is_feature_stats=True,
        text_feature_extraction: TextFeatureSpec = None,
        is_pre_model=True,
        is_model_training=True, model_run: ModelRun = None,
        is_post_model=True,
        is_model_weaknesses=True,
        plot=True,
        max_stats_col_width=30,
):
    """
        Perform end-to-end machine learning modeling on the provided data.

        This function takes in a data set and performs a comprehensive machine learning modeling process.
        It first identifies the problem type (either regression or classification) and the feature types
        (numeric, categorical, boolean, or text) based on metadata provided.
        Then, it selects the appropriate machine learning models to run, performs cross-validation (CV),
        and returns the best model based on comparison of the results.
        In addition, the function also provides feature statistics, pre-model analysis, post-model analysis,
        and identifies any weaknesses in the model.
        """
    # Tell a story
    story = """Welcome to MadCat, the comprehensive machine learning and data analysis solution!
    \nThis module is designed to streamline the entire process of data analysis and machine learning modeling, 
    \nfrom start to finish.
    \nTo learn more about MadCat, visit https://github.com/amito-ds/chester.
    \nMadCat performs all necessary pre-processing steps to get your data ready for modeling.
    \nIt then trains and tests multiple models, selecting the best one based on results.
    \nThe results are visually displayed for easy interpretation.\n"""

    if is_feature_stats:
        story += "The feature stats will be calculated to understand the data better.\n"
    if is_pre_model:
        story += "The pre model process will be performed to prepare the data for the model.\n"
    if data_spec.target_column is not None:
        story += "A model will be trained and tested using the given data.\n"
    if is_post_model:
        story += "Post model analysis will be performed to evaluate the model performance.\n"
    if is_model_weaknesses:
        story += "The weaknesses of the model will be analyzed to understand how to improve it.\n"
    if plot:
        story += "The results will be plotted for visualization.\n"
    story += "The maximum column width for stats display is set to {}.\n".format(max_stats_col_width)
    print(story)

    run_metadata_collector = {}

    # meta learn
    df = data_spec.df.sample(frac=1).reset_index(drop=True)
    data_info = DataInfo(data=df, target='target')
    data_info.calculate()

    if feature_types is not None:
        data_info = FeatureTypes(feature_types).update_data_info(data_info=data_info)
    target_column = data_info.target

    print(chapter_title("meta learn"))
    print(data_info)
    run_metadata_collector["data info"] = data_info
    ####################################################
    # Text handling
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

        # keep a copy of all text after cleaning
        text_cols = data_info.feature_types_val["text"]
        clean_text_df = pd.DataFrame()
        if len(text_cols) > 0:
            pd.options.mode.chained_assignment = None
            clean_text_df = data_info.data[text_cols]
            clean_text_df.rename(columns={col: "clean_" + col for col in clean_text_df.columns}, inplace=True)

        # preprocessing
        pp = PreprocessHandler(data_info=data_info, text_pre_process=text_pre_process)
        pp.transform()
        data_info_original = data_info  # in case you need it
        data_info = pp.data_info

        # data_info_text_cleaning for later text analysis
        if len(text_cols) > 0:
            clean_text_df = pd.concat([df, clean_text_df], axis=1)
        run_metadata_collector["data info"] = data_info
    ####################################################

    # Feat extract
    print(chapter_title('feature engineering'))
    if text_feature_extraction is not None:
        feat_hand = FeaturesHandler(data_info=data_info, text_feature_extraction=text_feature_extraction)
    else:
        feat_hand = FeaturesHandler(data_info=data_info)
    feature_types, final_df = feat_hand.transform()
    final_df[target_column] = data_info.data[data_info.target]

    run_metadata_collector["data info"] = data_info
    run_metadata_collector["features data"] = final_df
    ####################################################

    # Feat Stats
    data_info_num_stats = None
    if is_feature_stats:
        print(chapter_title('feature statistics'))
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

        max_stats_col_width = max_stats_col_width or 30
        if len(num_cols) > 0:
            print("Numerical Feature statistics")
            NumericStats(data_info_num_stats, max_print=max_stats_col_width).run(plot=plot_stats)
        print("Categorical Feature statistics")
        CategoricalStats(data_info, max_print=max_stats_col_width).run(plot=plot_stats)
        if len(text_cols) > 0:
            print("Text Feature statistics")
            data_info.data = clean_text_df
            TextStats(data_info).run()
    ####################################################

    # No target => no model! The story ends here
    if data_info.problem_type_val == "No target variable":
        return run_metadata_collector

    # Pre model
    if is_pre_model:
        print(chapter_title('model pre analysis'))
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
    # model
    if is_model_training:
        print(chapter_title("model training"))
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
        print(f"Best model: {type(model_results[1].model)}, with parameters:")
        for p in params:
            print(p.name, ":", p.value)

        run_metadata_collector["data info"] = data_info
        run_metadata_collector["features data"] = final_df
        run_metadata_collector["model"] = model
        run_metadata_collector["parameters"] = model_results[1]
        run_metadata_collector["model_results"] = model_results[0]
    ####################################################

    # post model
    if is_post_model and is_model_training:
        print(chapter_title('post model analysis'))
        post_model_analysis = PostModelAnalysis(cv_data, data_info, model=model_results[1])
        run_metadata_collector["post_model_analysis"] = post_model_analysis
        post_model_analysis.analyze()

        model_bootstrap = ModelBootstrap(cv_data, data_info, model=model_results[1])
        run_metadata_collector["model_bootstrap"] = model_bootstrap
        model_bootstrap.plot()
    ####################################################

    #  model weaknesses
    if is_model_weaknesses and is_model_training:
        print(chapter_title('model weaknesses'))
        model_weaknesses = ModelWeaknesses(cv_data, data_info, model=model_results[1])
        run_metadata_collector["model_weaknesses"] = model_weaknesses
        model_weaknesses.run()
    ####################################################

    return run_metadata_collector
