from itertools import chain

from chester.feature_stats.numeric_stats import NumericStats
from matplotlib import pyplot as plt

from chester.cleaning.cleaner_handler import CleanerHandler
from chester.feature_stats.categorical_stats import CategoricalStats
from chester.feature_stats.text_stats import TextStats
from chester.feature_stats.time_series_stats import TimeSeriesFeatureStatistics
from chester.features_engineering.features_handler import FeaturesHandler
from chester.features_engineering.time_series.ts_features_extraction import TimeSeriesFeaturesExtraction
from chester.model_monitor.error_prediction import ModelWeaknesses
from chester.model_monitor.model_boostrap import ModelBootstrap
from chester.model_training.models.chester_models.best_model import BestModel
from chester.post_model_analysis.post_model_analysis_class import PostModelAnalysis
from chester.pre_model_analysis.categorical import CategoricalPreModelAnalysis
from chester.pre_model_analysis.numerics import NumericPreModelAnalysis
from chester.pre_model_analysis.target import TargetPreModelAnalysis
from chester.pre_model_analysis.time_series import TimeSeriesPreModelAnalysis
from chester.preprocessing.preprocessor_handler import PreprocessHandler
from chester.run.chapter_titles import chapter_title
from chester.run.user_classes import Data, TextHandler, FeatureStats, ModelRun, TextFeatureExtraction, FeatureTypes, \
    TimeSeriesHandler, TextSummary, break_text_into_rows
from chester.text_stats_analysis.text_summary import get_summary
from chester.util import REPORT_PATH, ReportCollector
from chester.zero_break.problem_specification import DataInfo

import logging

logging.basicConfig(level=logging.WARNING)

from chester.model_training.data_preparation import CVData
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def run(
        data_spec: Data,
        feature_types: dict = None,
        text_handler: TextHandler = None, is_text_handler=True,
        text_summary: TextSummary = None,
        time_series_handler: TimeSeriesHandler = None, is_time_series_handler=True,
        feature_stats: FeatureStats = None, is_feature_stats=True,
        text_feature_extraction: TextFeatureExtraction = None, is_feature_extract=True,
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

    # reset the file
    rc = ReportCollector(REPORT_PATH)
    with open(REPORT_PATH, 'w') as f:
        pass
    chester_collector = {}

    # meta learn
    df = data_spec.df.sample(frac=1).reset_index(drop=True)
    data_info = DataInfo(data=df.sample(min(10000, len(df))), target=data_spec.target_column)
    data_info.calculate()
    data_info.data = df

    if feature_types is not None:
        data_info = FeatureTypes(feature_types).update_data_info(data_info=data_info)
    target_column = data_info.target

    print(chapter_title("meta learn"))
    print(data_info)
    chester_collector["data info"] = data_info

    rc.save_text(text="Full report to analyze my data:\n")
    rc.save_object(obj=data_info, text="Data information:")
    ####################################################
    # Text handling
    # Summary, if asked
    text_cols = data_info.feature_types_val["text"]
    if text_summary is None:
        text_summary = TextSummary(is_summary=True)
    if len(text_cols) > 0 and text_summary is not None:
        if text_summary.is_summary:
            chester_collector["raw text summary"] = get_summary(df=df, text_columns=text_cols,
                                                                text_summary=text_summary)

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
        chester_collector["data info"] = data_info
    ####################################################

    ####################################################
    # Time series handling
    # is_time_series_handler
    ####################################################
    # Feat extract
    if is_feature_extract:
        print(chapter_title('feature engineering'))
        rc.save_text(text="features engineering process for the data:")
        # Handle TS
        orig_di = data_info
        if is_time_series_handler:
            if time_series_handler is None:
                time_series_handler = TimeSeriesHandler()
            time_cols = data_info.feature_types_val["time"]
            if len(time_cols) > 0:
                ts_fe = TimeSeriesFeaturesExtraction(data_info=data_info, time_series_handler=time_series_handler)
                ts_fe.run()
                ts_fe.data_info.data = ts_fe.data_info.data. \
                                           loc[:, ~ts_fe.data_info.data.columns.duplicated()]  # drop col dups
                data_info = ts_fe.data_info
                time_series_handler = ts_fe.time_series_handler
                # orig_di = data_info

        # Continue the rest
        feat_hand = FeaturesHandler(data_info=data_info)
        if text_feature_extraction is not None:
            feat_hand.text_feature_extraction = text_feature_extraction
        feature_types, final_df = feat_hand.transform()
        if target_column is not None:
            final_df[target_column] = data_info.data[data_info.target]
        final_df = final_df.loc[:, ~final_df.columns.duplicated()]
        data_info.data = data_info.data.loc[:, ~data_info.data.columns.duplicated()]

        chester_collector["data info"] = data_info
        chester_collector["features data"] = final_df
    ####################################################
    # Feat Stats
    data_info_num_stats = None
    if is_feature_stats:
        rc.save_text(
            text="features statistics for the data. Analyzed by groups (text, numeric, categorical) if exists:")
        print(chapter_title('feature statistics'))
        if feature_stats is None:
            feature_stats = FeatureStats()
        sample_obs = feature_stats.sample_obs
        # sample data for stats
        final_df = final_df.loc[:, ~final_df.columns.duplicated()].sample(min(sample_obs, len(final_df)))
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
            title = "\nNumerical Feature statistics:"
            print(title)
            rc.save_text(title)
            NumericStats(data_info_num_stats, max_print=max_stats_col_width).run(plot=plot_stats)
        title = "\nCategorical Feature statistics:"
        print(title)
        rc.save_text(title)

        CategoricalStats(data_info, max_print=max_stats_col_width).run(plot=plot_stats)
        if len(text_cols) > 0:
            print("Text Feature Statistics")
            orig_df = data_info.data
            data_info.data = clean_text_df
            if text_feature_extraction is None:
                text_feature_extraction = TextFeatureExtraction()
            TextStats(data_info, text_spec=text_feature_extraction, chester_collector=chester_collector).run()
            data_info.data = orig_df
            if "raw text summary" in chester_collector is not None:
                print(f"Text Columns Summary:")
                try:
                    for col in text_cols:
                        print("\t", chester_collector["raw text summary"][col])
                except:
                    print("\t", chester_collector["raw text summary"])

        all_features = list(set(chain.from_iterable(list(data_info.feature_types_val.values()))))
        ts_cols = [feat for feat in all_features if feat.startswith("ts_")]
        if len(ts_cols) > 0:
            print("Time Series Feature Statistics")
            TimeSeriesFeatureStatistics(data_info=data_info, ts_cols=ts_cols,
                                        time_series_handler=time_series_handler).run(plot=True)
    ####################################################
    # No target => no model! The story ends here
    if data_info.problem_type_val == "No target variable":
        return chester_collector

    # Pre model
    if is_pre_model:
        final_df = final_df.loc[:, ~final_df.columns.duplicated()]
        rc.save_text("** model pre analysis report:")
        print(chapter_title('model pre analysis'))
        # label stats
        print("Label pre model analysis:")
        TargetPreModelAnalysis(data_info=data_info, time_series_handler=time_series_handler).run(plot)
        TimeSeriesPreModelAnalysis(data_info=data_info, time_series_handler=time_series_handler).run()
        # num, cat pre model
        if data_info_num_stats is None:
            data_info_num_stats = DataInfo(data=final_df, target=target_column)
            data_info_num_stats.calculate()
        NumericPreModelAnalysis(data_info_num_stats).run(plot)
        plt.close()
        # cat if found any
        cat_not_ts = data_info.feature_types_val["categorical"]
        if len(cat_not_ts) > 0:
            print("Categorical pre model analysis:")
            CategoricalPreModelAnalysis(data_info).run(plot)
    ####################################################
    # model
    if not is_model_training:
        return chester_collector
    if is_model_training:
        print(chapter_title("model training"))
        rc.save_text("Training models and choosing the best one")
        # drop col dups
        data_info.feature_types_val["numeric"] = list(set(data_info.feature_types_val["numeric"]))
        data_info.feature_types_val["categorical"] = list(set(data_info.feature_types_val["categorical"]))
        # encode if needed
        if data_info.problem_type_val in ["Binary classification", "Multiclass classification"]:
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
        try:
            print(f"Best model: {type(model_results[1].model)}, with parameters:")
            for p in params:
                print(p.name, ":", p.value)
        except:
            pass

        chester_collector["data info"] = data_info
        chester_collector["features data"] = final_df
        chester_collector["model"] = model_results[1]
        chester_collector["parameters"] = params
        chester_collector["model_results"] = model_results[0]
    ####################################################

    # post model
    is_baseline = type(model_results[1]).__name__ == "BaselineModel"
    if is_baseline:
        print("Best model is a simple baseline")
        return chester_collector
    if is_post_model and is_model_training:
        rc.save_text("\nPost model analysis - analyzing results of the chosen model: ")
        print(chapter_title('post model analysis'))
        post_model_analysis = PostModelAnalysis(cv_data, data_info, model=model_results[1])
        chester_collector["post_model_analysis"] = post_model_analysis
        post_model_analysis.analyze()

        model_bootstrap = ModelBootstrap(cv_data, data_info, model=model_results[1])
        chester_collector["model_bootstrap"] = model_bootstrap
        model_bootstrap.plot()
    ####################################################

    #  model weaknesses
    if is_model_weaknesses and is_model_training:
        rc.save_text("Trying to find weaknesses in the model by training models on the error:")
        print(chapter_title('model weaknesses'))
        model_weaknesses = ModelWeaknesses(cv_data, data_info, model=model_results[1])
        chester_collector["model_weaknesses"] = model_weaknesses
        model_weaknesses.run()
    ####################################################

    return chester_collector
