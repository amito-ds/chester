import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

from chester.cleaning.cleaner_handler import CleanerHandler
from chester.data_loader.webtext_data import load_data_pirates, load_data_king_arthur, load_data_chat_logs
from chester.feature_stats.categorical_stats import CategoricalStats
from chester.feature_stats.numeric_stats import NumericStats
from chester.features_engineering.features_handler import FeaturesHandler
from chester.model_training.data_preparation import CVData
from chester.pre_model_analysis.categorical import CategoricPreModelAnalysis
from chester.pre_model_analysis.numerics import NumericPreModelAnalysis
from chester.preprocessing.preprocessor_handler import PreprocessHandler
from chester.zero_break.problem_specification import DataInfo

target_column = 'target'
################################################################################################
# df1 = load_data_pirates().assign(target='pirate').sample(100, replace=True)
# df2 = load_data_king_arthur().assign(target='arthur').sample(100, replace=True)
# df3 = load_data_chat_logs().assign(target='chat').sample(100, replace=True)
# df = pd.concat([df1, df2
#                    , df3
#                 ])
################################################################################################


#

################################################################################################
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# dataset = pd.read_csv(url, names=names)
# dataset.rename(columns={'class': 'target'}, inplace=True)
# df = dataset.sample(frac=1).reset_index(drop=True)
# df['target'] = df['target'].apply(lambda x: 0 if "Iris-setos" in x else 1) # can do with or without
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
# df['target'] = "category: " + df['target'].astype(str)
################################################################################################


################################################################################################
# categorical features
# import seaborn as sns
# df = sns.load_dataset("tips")
# df.rename(columns={'tip': target_column}, inplace=True)
################################################################################################

################################################################################################
# categorical
import seaborn as sns
df = sns.load_dataset("titanic")
df.rename(columns={'survived': target_column}, inplace=True)
df.drop(columns=['alive'], inplace=True)
###############################################################################################


# Print the first 5 rows of the dataframe

# # calc data into
df = df.sample(frac=1).reset_index(drop=True)
data_info = DataInfo(data=df, target='target')
data_info.calculate()
print(data_info)

# clean
cleaner = CleanerHandler(data_info)
cleaner.transform()
data_info = cleaner.data_info

pp = PreprocessHandler(data_info)
pp.transform()
data_info = pp.data_info

# extract features
feat_hand = FeaturesHandler(data_info)
feature_types, final_df = feat_hand.transform()
final_df[target_column] = data_info.data[data_info.target]

########## code for stats and PMA ################
data_info_num_stats = DataInfo(data=final_df, target='target')
data_info_num_stats.calculate()

# pma
# CategoricPreModelAnalysis(data_info).analyze_pvalue()
CategoricPreModelAnalysis(data_info).partial_plot()
# NumericPreModelAnalysis(data_info_num_stats).analyze_pvalue()
# print(NumericPreModelAnalysis(data_info_num_stats).partial_plot())
# NumericStats(data_info_num_stats).run()
# CategoricalStats(data_info).run()
########## code for stats ################

#### model
# encode labels
# label_encoder = LabelEncoder()
# final_df[target_column] = label_encoder.fit_transform(final_df[target_column])
# cv_data = CVData(train_data=final_df, test_data=None, target_column='target', split_data=True)