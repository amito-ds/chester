from collections import Counter

import pandas as pd
from flatbuffers.builder import np
from sklearn.datasets import fetch_openml, fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder

from chester.cleaning.cleaner_handler import CleanerHandler
from chester.data_loader.webtext_data import load_data_pirates, load_data_king_arthur, load_data_chat_logs
from chester.feature_stats.categorical_stats import CategoricalStats
from chester.feature_stats.numeric_stats import NumericStats
from chester.feature_stats.text_stats import TextStats
from chester.features_engineering.features_handler import FeaturesHandler
from chester.model_monitor.mm_bootstrap import ModelBootstrap
from chester.model_monitor.mm_weaknesses import ModelWeaknesses
from chester.model_training.data_preparation import CVData
from chester.model_training.models.chester_models.best_catboost import CatboostModel
from chester.model_training.models.chester_models.best_linear_regression import LinearRegressionModel
from chester.model_training.models.chester_models.best_logistic_regression import LogisticRegressionModel
from chester.model_training.models.chester_models.best_model import BestModel
from chester.post_model_analysis.post_model_analysis_class import PostModelAnalysis
from chester.pre_model_analysis.categorical import CategoricalPreModelAnalysis
from chester.pre_model_analysis.numerics import NumericPreModelAnalysis
from chester.pre_model_analysis.target import TargetPreModelAnalysis
from chester.preprocessing.preprocessor_handler import PreprocessHandler
from chester.run.full_run import run_madcat
from chester.run.user_classes import Data
from chester.zero_break.problem_specification import DataInfo

import matplotlib

# matplotlib.use("Agg")

matplotlib.use('TkAgg')
target_column = 'target'


################################################################################################
df1 = load_data_pirates().assign(target='pirate').sample(100, replace=True)
df2 = load_data_king_arthur().assign(target='arthur').sample(100, replace=True)
df3 = load_data_chat_logs().assign(target='chat').sample(100, replace=True)
df = pd.concat([df1, df2
                # , df3
                ])
# df['text_trimmed'] = df['text'].apply(lambda x: x[:100])
# df.rename(columns={'text': 'text_a'}, inplace=True)
################################################################################################


################################################################################################
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# dataset = pd.read_csv(url, names=names)
# dataset.rename(columns={'class': 'target'}, inplace=True)
# df = dataset.sample(frac=1).reset_index(drop=True)
# df['target'] = df['target'].apply(lambda x: 0 if "Iris-setos" in x else 1)  # can do with or without
###############################################################################################

###############################################################################################
# Load the Boston Housing dataset. categorical
# boston = fetch_openml(name='boston', version=1)
# df = pd.DataFrame(boston.data, columns=boston.feature_names)
# df['target'] = boston.target
###############################################################################################

###############################################################################################
# df = pd.read_csv("chester/model_training/models/chester_models/day.csv")
# df.rename(columns={'cnt': 'target'}, inplace=True)
###############################################################################################

################################################################################################
# from sklearn import datasets
# digits = datasets.load_digits()
# X = digits.images.reshape((len(digits.images), -1))
# df = pd.DataFrame(X)
# df.rename(columns={col: "feat_" + str(col) for col in df.columns}, inplace=True)
# df['target'] = digits.target
# df['target'] = "c_ " + df['target'].astype(str)
################################################################################################


################################################################################################
# categorical features
# import seaborn as sns
# df = sns.load_dataset("tips")
# df.rename(columns={'tip': target_column}, inplace=True)
################################################################################################

################################################################################################
# categorical
# import seaborn as sns
# df = sns.load_dataset("titanic")
# df.rename(columns={'survived': target_column}, inplace=True)
# df.drop(columns=['alive'], inplace=True)
###############################################################################################

###############################################################################################

def generate_data(n_features, n_rows, target_type='binary'):
    if target_type == 'binary':
        # Create binary target column
        target = np.random.choice(['yes', 'no'], size=n_rows)
    elif target_type == 'multiclass':
        # Create multiclass target column
        target = np.random.choice(['class_1', 'class_2', 'class_3', 'class_4',
                                   'class_5', 'class_6', 'class_7', 'class_8',
                                   'class_9', 'class_10', 'class_11', 'class_12'], size=n_rows)
    else:
        raise ValueError("Invalid target_type. Must be either 'binary' or 'multiclass'.")

    # Create feature categorical columns
    features = {}
    for i in range(n_features):
        feature = np.random.choice(['A', 'B', 'C', 'D'], size=n_rows)
        features[f'feature_{i}'] = feature

    # Create pandas DataFrame
    df = pd.DataFrame(features)
    df['target'] = target

    return df


# df = generate_data(20, 1000, target_type='binary')
# df = generate_data(30, 100, target_type='multiclass')
###############################################################################################

## vlad
def load_vlad():
    df = pd.read_csv("chester/model_training/models/chester_models/data.csv")
    df.rename(columns={'TOTAL_BET_AMOUNT': 'target'}, inplace=True)
    df['target'] = 1 * (df['REVENUE'] > 0.00001) + 1 * (df['REVENUE'] > 2)
    df.drop(
        columns=['REVENUE', 'Unnamed: 0', 'PLAYER_ID', 'MEDIAN_BET', 'SESSION_MINS', 'SPINS_COMPLETED', 'SPINS_STARTED',
                 'TOTAL_SPIN_LENGTH'], inplace=True)

    ## sample
    class_0 = df[df['target'] == 0].sample(5000)
    class_1 = df[df['target'] == 1]
    df = pd.concat([class_0, class_1])
    return df


def load_ex1():
    data = pd.read_csv("chester/model_training/models/chester_models/lead_df_2023-01-23.csv")
    data.rename(columns={'lawer_conversion_label': target_column}, inplace=True)
    data.drop(columns=["LEAD_ID"], inplace=True)
    return data


def load_ex2():
    newsgroups_train = fetch_20newsgroups(subset='train')
    df = pd.DataFrame(newsgroups_train.data, columns=['text'])
    y = newsgroups_train.target
    df['target'] = y
    category_counts = Counter(y)
    top_3_categories = category_counts.most_common(4)
    top_3_categories = [cat for cat, count in top_3_categories]
    df = df[df.target.isin(top_3_categories)].sample(1500)
    df['target'] = "c_ " + df['target'].astype(str)
    return df


import ml_datasets


def load_ex3():
    train_data, _ = ml_datasets.dbpedia()
    train_data = pd.DataFrame(train_data, columns=["text", "target"])
    train_data['target'] = "c_: " + train_data['target'].astype(str)
    return train_data


def load_ex4():
    train_data, _ = ml_datasets.cmu()
    train_data = pd.DataFrame(train_data, columns=["text", "target"])
    train_data['target'] = "category: " + train_data['target'].astype(str)
    train_data = train_data[train_data['target'].isin(train_data['target'].value_counts()[:5].index)]
    return train_data


def load_ex5():
    train_data, _ = ml_datasets.quora_questions()
    train_data = [(a, b, c) for ((a, b), c) in train_data]
    train_data = pd.DataFrame(train_data, columns=["text1", "text2", "target"])
    return train_data


# load data
# df = load_vlad()
# df = load_ex1()
# df = load_ex2()
# df = load_ex3().sample(1000)
# df = load_ex4().sample(1000)
# df = load_ex5().sample(900)


run_madcat(Data(df=df, target_column='target'),
           is_feature_stats=False,
           is_pre_model=False)

#
# # # calc data into
# print("XXXXXXXXXXXXXXXXXXXXXXXXData infoXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
# df = df.sample(frac=1).reset_index(drop=True)
# data_info = DataInfo(data=df, target='target')
# data_info.calculate()
# print(data_info)
#
# # clean
# print("XXXXXXXXXXXXXXXXXXXXXXXXCleanXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
# cleaner = CleanerHandler(data_info)
# cleaner.transform()
# data_info = cleaner.data_info
#
# # keep a copy of all text after cleaning
# text_cols = data_info.feature_types_val["text"]
# clean_text_df = pd.DataFrame()
# if len(text_cols) > 0:
#     pd.options.mode.chained_assignment = None
#     clean_text_df = data_info.data[text_cols]
#     clean_text_df.rename(columns={col: "clean_" + col for col in clean_text_df.columns}, inplace=True)
#
# # PP
# print("XXXXXXXXXXXXXXXXXXXXXXXXPPXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
# pp = PreprocessHandler(data_info)
# pp.transform()
# data_info_original = data_info
# data_info = pp.data_info
#
# # data_info_text_cleaning
# if len(text_cols) > 0:
#     clean_text_df = pd.concat([df, clean_text_df], axis=1)
#
# # # extract features
# print("XXXXXXXXXXXXXXXXXXXXXXXXExtract featuresXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
# feat_hand = FeaturesHandler(data_info)
# feature_types, final_df = feat_hand.transform()
# final_df[target_column] = data_info.data[data_info.target]
#
# #### stats: start
# print("XXXXXXXXXXXXXXXXXXXXXXXXFeature StatsXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
# data_info_num_stats = DataInfo(data=final_df, target=target_column)
# data_info_num_stats.calculate()
# TargetPreModelAnalysis(data_info).run()
# print("Numerical Feature statistics")
# NumericStats(data_info_num_stats).run()
# print("Categorical Feature statistics")
# CategoricalStats(data_info).run()
# print("Text Feature statistics")
# data_info.data = clean_text_df
# TextStats(data_info).run()
# ### stats: end
#
# ########## code for stats and PMA ################
# # pma
# print("XXXXXXXXXXXXXXXXXXXXXXXXPre model analysisXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
# NumericPreModelAnalysis(data_info_num_stats).run()
# data_info.data = df
# CategoricalPreModelAnalysis(data_info).run()
# # ########## code for stats ################
#
# # #################################### model####################################
# # encode labels if needed (for classification problem only)
# print("XXXXXXXXXXXXXXXXXXXXXXXXModel runXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
# if data_info.problem_type_val in ["Binary classification", "Multiclass classification"]:
#     print("Encoding target")
#     label_encoder = LabelEncoder()
#     final_df[target_column] = label_encoder.fit_transform(final_df[target_column])
# #
# # # Run the model
# cv_data = CVData(train_data=final_df, test_data=None, target_column='target', split_data=True)
# data_info.feature_types_val = feature_types
# model = BestModel(data_info=data_info, cv_data=cv_data, num_models_to_compare=3)
# model_results = model.get_best_model()  # returns resultf of the best baseline model
# params = model_results[1].get_params()
# print(f"Best model: {type(model_results[1])}, with parameters:")
# for p in params:
#     print(p.name, ":", p.value)
# ################################### model####################################
#
# #################################### PMA####################################
# print("XXXXXXXXXXXXXXXXXXXXXXXXPost model analysisXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
# PostModelAnalysis(cv_data, data_info, model=model_results[1]).analyze()
# ModelBootstrap(cv_data, data_info, model=model_results[1]).plot()
# #################################### PMA ####################################
#
#
# #################################### monitor ####################################
# print("XXXXXXXXXXXXXXXXXXXXXXXXMoitorXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
# model_weaknesses = ModelWeaknesses(cv_data, data_info, model=model_results[1])
# model_weaknesses.run()
# #################################### monitor####################################
