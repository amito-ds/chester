import pandas as pd
from sklearn.preprocessing import LabelEncoder

from mdoel_training.best_model import ModelCycle
from mdoel_training.data_preparation import CVData

#
# #
# df1 = load_data_pirates().assign(target='chat_logs')
# df2 = load_data_king_arthur().assign(target='pirates')
# df = pd.concat([df1, df2])
# #
# # # # Clean the text column
# get_sw = get_stopwords()
# df['text'] = df['text'].apply(lambda x: clean_text(x,
#                                                    remove_stopwords_flag=True,
#                                                    stopwords=get_sw))
# #
# # # preprocess the text column
# df['clean_text'] = df['text'].apply(lambda x:
#                                     preprocess_text(x, stemmer=get_stemmer('porter'), stem_flag=True))
# #
# # #
# train_embedding, test_embedding = get_embeddings(training_data=df, corex=True, tfidf=False, bow=False, corex_dim=10)
#
# #
# # print(train_embedding.shape)
# # print(test_embedding.shape)
# #
# train_embedding.to_csv("train_embedding.csv")
# test_embedding.to_csv("test_embedding.csv")

train_embedding = pd.read_csv("train_embedding.csv")
test_embedding = pd.read_csv("test_embedding.csv")
train_embedding = train_embedding.drop(train_embedding.columns[0], axis=1)
test_embedding = test_embedding.drop(test_embedding.columns[0], axis=1)

target_col = 'target'
label_encoder = LabelEncoder()
train_embedding[target_col] = label_encoder.fit_transform(train_embedding[target_col])
test_embedding[target_col] = label_encoder.transform(test_embedding[target_col])

# Create a CVData object
cv_data = CVData(train_data=train_embedding, test_data=test_embedding, folds=2)
best_model = ModelCycle(cv_data=cv_data, target_col='target').get_best_model()

print(best_model.predict())
# best_params, best_score = \
#     Optimizer(logistic_regression_hp,
#               logistic_regression_best_practice_hp(),
#               "random search",
#               ModelInput(cv_data, [], target_col='target'), model_type="logistic regression class").random_search(3)
# results, model, parameters = lgbm_with_outputs(cv_data, best_params, target_col=target_col)

# print(train_embedding['Unnamed: 0'])
# lgbm_grid_search(cv_data=cv_data, parameters=hp_space(), target_col=target_col)

# # # # # Organize results
# organized_results = organize_results(best_model.results)
# print(organized_results.shape)
# # # #
# analyze_results(organized_results, best_model.parameters)
# analyze_model(best_model.model, cv_data, target_label='target')
