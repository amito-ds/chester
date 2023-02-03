import pandas as pd
from sklearn.preprocessing import LabelEncoder

from chester.cleaning.cleaner_handler import CleanerHandler
from chester.feature_stats.numeric_stats import NumericStats
from chester.features_engineering.features_handler import FeaturesHandler
from chester.model_training.data_preparation import CVData
from chester.preprocessing.preprocessor_handler import PreprocessHandler
from chester.utils.df_utils import format_df
from chester.zero_break.problem_specification import DataInfo


class FeatureStats:
    def __init__(self, data_info: DataInfo):
        self.data_info = data_info

    def calculate_stats(self):
        pass


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
# Load the Boston Housing dataset
# boston = fetch_openml(name='boston', version=1)
# df = pd.DataFrame(boston.data, columns=boston.feature_names)
# df['target'] = boston.target
###############################################################################################

###############################################################################################
df = pd.read_csv("chester/model_training/models/chester_models/day.csv")
df.rename(columns={'cnt': 'target'}, inplace=True)
###############################################################################################

# newsgroups_train = fetch_20newsgroups(subset='train')
# df = pd.DataFrame(newsgroups_train.data, columns=['text'])
# y = newsgroups_train.target
# df['target'] = y
# category_counts = Counter(y)
# top_3_categories = category_counts.most_common(3)
# top_3_categories = [cat for cat, count in top_3_categories]
# df = df[df.target.isin(top_3_categories)].sample(1000)
# df['target'] = "category: " + df['target'].astype(str)

# import pandas as pd
################################################################################################
# from sklearn import datasets
# digits = datasets.load_digits()
# X = digits.images.reshape((len(digits.images), -1))
# df = pd.DataFrame(X)
# df.rename(columns={col: "feat_" + str(col) for col in df.columns}, inplace=True)
# df['target'] = digits.target
# df['target'] = "category: " + df['target'].astype(str)
################################################################################################


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

## stats
NumericStats(data_info).calculate_stats()
# # extract features
# feat_hand = FeaturesHandler(data_info)
# feature_types, final_df = feat_hand.transform()
# final_df[target_column] = data_info.data[data_info.target]

#
# # # label transformer
# label_encoder = LabelEncoder()
# final_df[target_column] = label_encoder.fit_transform(final_df[target_column])
# # print(final_df)
# #
# cv_data = CVData(train_data=final_df, test_data=None, target_column='target', split_data=True)
