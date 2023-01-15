import pandas as pd

from cleaning.cleaning import clean_text
from data_loader.webtext_data import load_data_pirates, load_data_king_arthur
from features_engineering.fe_main import get_embeddings
from mdoel_training.best_model import ModelCycle
from mdoel_training.data_preparation import CVData
from mdoel_training.model_utils import organize_results, analyze_results
from model_analyzer.model_analysis import analyze_model
from preprocessing.preprocessing import preprocess_text, get_stemmer
from util import get_stopwords





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
# # #
# # #
# train_embedding, test_embedding = get_embeddings(training_data=df, corex=True, tfidf=True, bow=True, corex_dim=5)
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


# Create a CVData object
cv_data = CVData(train_data=train_embedding, test_data=test_embedding)
best_model = ModelCycle(cv_data=cv_data, target_col='target').get_best_model()

# print(train_embedding['Unnamed: 0'])

# # # # # Organize results
organized_results = organize_results(best_model.results)
# print(organized_results.shape)
# # # #
analyze_results(organized_results, best_model.parameters)
analyze_model(best_model.model, cv_data, target_label='target')
