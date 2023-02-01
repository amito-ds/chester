from collections import Counter

import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder

from chester.cleaning.cleaner_handler import CleanerHandler
from chester.features_engineering.features_handler import FeaturesHandler
from chester.model_training.data_preparation import CVData
from chester.model_training.models.chester_models import best_catboost
from chester.model_training.models.chester_models import best_logistic_regression
from chester.model_training.models.chester_models.base_model import BaseModel
from chester.model_training.models.chester_models.best_baseline_model import BaselineModel
from chester.model_training.models.chester_models.best_linear_regression import LinearRegressionModel
from chester.model_training.models.chester_models.catboost.catboost_utils import compare_best_models
from chester.post_model_analysis.post_model_analysis_class import PostModelAnalysis
from chester.preprocessing.preprocessor_handler import PreprocessHandler
from chester.zero_break.problem_specification import DataInfo


class BestModel(BaseModel):
    def __init__(self, data_info: DataInfo, cv_data: CVData, num_models_to_compare=15):
        super().__init__(data_info, cv_data, num_models_to_compare)

    def get_best_model(self):
        models_type = self.data_info.model_selection_val
        models = []
        for model_type in models_type:
            if 'baseline' in model_type:
                base_res, model = BaselineModel(data_info=self.data_info,
                                                cv_data=self.cv_data,
                                                num_models_to_compare=self.num_models_to_compare
                                                ).get_best_model()
                models.append((base_res, model))
            elif 'logistic' in model_type:
                base_res, model = best_logistic_regression.LogisticRegressionModel(
                    data_info=self.data_info,
                    cv_data=self.cv_data,
                    num_models_to_compare=self.num_models_to_compare
                ).get_best_model()
                models.append((base_res, model))
            elif 'catboost' in model_type:
                base_res, model = best_catboost.CatboostModel(
                    data_info=self.data_info,
                    cv_data=self.cv_data,
                    num_models_to_compare=self.num_models_to_compare
                ).get_best_model()
                models.append((base_res, model))
            elif 'linear' in model_type:
                base_res, model = LinearRegressionModel(
                    data_info=self.data_info,
                    cv_data=self.cv_data,
                    num_models_to_compare=self.num_models_to_compare
                ).get_best_model()
                print("linear base res")
                print(base_res)
                models.append((base_res, model))
        if models is None:
            return None
        else:
            print("Finding the best model out of model types ran")
            best = compare_best_models(models)
            return best


# df1 = load_data_pirates().assign(target='pirate').sample(100, replace=True)
# df2 = load_data_king_arthur().assign(target='arthur').sample(100, replace=True)
# df3 = load_data_chat_logs().assign(target='chat').sample(100, replace=True)
# df = pd.concat([df1, df2
#                    # , df3
#                 ])
target_column = 'target'
#

# target_column = 'target'
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# dataset = pd.read_csv(url, names=names)
# dataset.rename(columns={'class': 'target'}, inplace=True)

# df = dataset.sample(frac=1).reset_index(drop=True)
# df['target'] = df['target'].apply(lambda x: 0 if "Iris-setos" in x else 1)


# Load the Boston Housing dataset
# boston = fetch_openml(name='boston', version=1)
# df = pd.DataFrame(boston.data, columns=boston.feature_names)
# df['target'] = boston.target
df = pd.read_csv("chester/model_training/models/chester_models/day.csv")
df.rename(columns={'cnt': 'target'}, inplace=True)

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
# from sklearn import datasets

# Load the digits dataset from scikit-learn
# digits = datasets.load_digits()

# Flatten the images from 8x8 arrays to 64-dimensional vectors
# X = digits.images.reshape((len(digits.images), -1))

# Create a dataframe with the features and target
# df = pd.DataFrame(X)
# df.rename(columns={col: "feat_" + str(col) for col in df.columns}, inplace=True)
# df['target'] = digits.target
# df['target'] = "category: " + df['target'].astype(str)

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

# # extract features
feat_hand = FeaturesHandler(data_info)
feature_types, final_df = feat_hand.transform()
final_df[target_column] = data_info.data[data_info.target]

#
# # # label transformer
label_encoder = LabelEncoder()
final_df[target_column] = label_encoder.fit_transform(final_df[target_column])
# # print(final_df)
# #
cv_data = CVData(train_data=final_df, test_data=None, target_column='target', split_data=True)
model = BestModel(data_info=data_info, cv_data=cv_data, num_models_to_compare=3)
# model = BestModel(data_info=data_info, cv_data=cv_data, num_models_to_compare=15)
# #
model_results = model.get_best_model()  # returns resultf of the best baseline model
# print(model_results[0][['type', 'fold', 'mean_squared_error', 'mean_absolute_error']])
# print(model_results[0])
# params = model_results[1].get_params()
# for p in params:
#     print(p.name, p.value)
    # #
PostModelAnalysis(cv_data, data_info, model=model_results[1]).analyze()
# analyze_model(model_results[0], cv_data, target_label=target_column)
