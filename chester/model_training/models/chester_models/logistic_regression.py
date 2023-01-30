import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from chester.data_loader.webtext_data import load_data_pirates, load_data_king_arthur, load_data_chat_logs
from chester.features_engineering.features_handler import FeaturesHandler
from chester.model_training.data_preparation import CVData
from chester.model_training.models.chester_models.base_model import BaseModel
from chester.model_training.models.chester_models.base_model_utils import calculate_metrics_scores
from chester.model_training.models.chester_models.logistic_regression.logistic_regression_utils import \
    generate_logistic_regression_configs, predict_logistic_regression, logistic_regression_with_outputs, compare_models
from chester.model_training.models.chester_models.logistic_regression.logistic_regression_utils import \
    train_logistic_regression
from chester.zero_break.problem_specification import DataInfo


class LogisticRegressionModel(BaseModel):
    def __init__(self, data_info: DataInfo, cv_data: CVData, num_models_to_compare=10):
        super().__init__(data_info, cv_data, num_models_to_compare)
        self.hp_list = generate_logistic_regression_configs(self.num_models_to_compare)

    def train_single_model(self, param: list):
        results = []
        metrics_func = self.get_metrics_functions()
        for i, X_train, y_train, X_test, y_test in enumerate(self.cv_data.format_splits()):
            model = train_logistic_regression(X_train=X_train, y_train=y_train, parameters=param)
            prediction = predict_logistic_regression(model, X_test)
            prediction_train = predict_logistic_regression(model, X_train)
            scores = calculate_metrics_scores(y_test, prediction, metrics_func, self.data.problem_type_val)
            results.append({'type': 'test', 'fold': i, **scores})
            scores = calculate_metrics_scores(y_train, prediction_train, metrics_func, self.data.problem_type_val)
            results.append({'type': 'train', 'fold': i, **scores})
        return results

    def get_best_model(self):
        models = self.data.model_selection_val
        metrics = self.get_metrics_functions()
        if models is None:
            return None
        else:
            lr_models = [model for model in models if "logistic" in model]
            # print("lr_models", lr_models)
            if len(lr_models) == 0:
                return None
            else:
                results = []
                for _ in lr_models:
                    for params in self.hp_list:
                        base_res, model = logistic_regression_with_outputs(
                            cv_data=self.cv_data, target_col=self.cv_data.target_column,
                            parameters=params, metrics=metrics, problem_type=self.data.problem_type_val)
                        results.append((base_res, model))
                best = compare_models(results)
                return best

    def calc_model_score(self, model):
        # calculate the score of a given baseline model
        pass


#
# df1 = load_data_pirates().assign(target='pirate')
# df2 = load_data_king_arthur().assign(target='arthur')
# df3 = load_data_chat_logs().assign(target='chat')
# df = pd.concat([df1, df2, df3])
target_column = 'target'
#
# df = df.sample(frac=1).reset_index(drop=True)
# df["number"] = np.random.uniform(0, 1, df.shape[0])
# df["categ"] = 'aaa'
# df["booly"] = True
# df['target'] = df['target'].apply(lambda x: 0 if "pirate" in x or "arthur" in x else 1)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)
dataset.rename(columns={'class': 'target'}, inplace=True)

# df.drop(columns='text', inplace=True)
df = dataset.sample(frac=1).reset_index(drop=True)
df['target'] = df['target'].apply(lambda x: 0 if "Iris-setos" in x else 1)

# print(df)
#
# # calc data into
data_info = DataInfo(data=df, target='target')
data_info.calculate()
print(data_info)
#
# # extract features
feat_hand = FeaturesHandler(data_info)
feature_types, final_df = feat_hand.transform()
final_df[target_column] = data_info.data[data_info.target]

#
# # label transformer
label_encoder = LabelEncoder()
final_df[target_column] = label_encoder.fit_transform(final_df[target_column])
# print(final_df)
#
cv_data = CVData(train_data=final_df, test_data=None, target_column='target')
lr_model = LogisticRegressionModel(data_info=data_info, cv_data=cv_data)
#
model_results = lr_model.get_best_model()  # returns resultf of the best baseline model
# print(model_results[0].drop(columns=['type', 'fold']))
params = model_results[1].get_params()
for p in params:
    print(p.name, p.value)
#
