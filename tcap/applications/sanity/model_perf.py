# THIS EXAMPLE WORKS SO DONT CHANGE THE CODE!!!!
# it should choose lgbm and show the shap and FE

# 1. lgbm with value: 0.56
# 2. logistic regression with value: 0.896
# 3. baseline with value: 0.729
# The best models for recall_score metric are:
# 1. lgbm with value: 0.0
# 2. logistic regression with value: 0.889
# 3. baseline with value: 0.661
# The best models for f1_score metric are:
# 1. lgbm with value: 0.0
# 2. logistic regression with value: 0.882
# 3. baseline with value: 0.677

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from cleaning.cleaning_func import clean_text
from data_loader.webtext_data import load_data_pirates, load_data_king_arthur
from features_engineering.feature_main import get_embeddings
from tcap.model_training.best_model import ModelCycle
from tcap.model_training.data_preparation import CVData
from tcap.model_training.model_utils import  analyze_results
from tcap.model_analyzer.model_analysis import analyze_model
from preprocessing.preprocessing_func import preprocess_text, get_stemmer
from tcap.util import get_stopwords

#
# #
df1 = load_data_pirates().assign(target='chat_logs')
df2 = load_data_king_arthur().assign(target='pirates')
df = pd.concat([df1, df2])
# #
# # # # Clean the text column
get_sw = get_stopwords()
df['text'] = df['text'].apply(lambda x: clean_text(x,
                                                   remove_stopwords_flag=True,
                                                   stopwords=get_sw))
# #
# # # preprocess the text column
df['text'] = df['text'].apply(lambda x:
                                    preprocess_text(x, stemmer=get_stemmer('porter'), stem_flag=True))

train_embedding, test_embedding = get_embeddings(
    training_data=df,
    corex=True, corex_dim=80,
    tfidf=True, tfidf_dim=90,
    bow=True, bow_dim=100)


target_col = 'target'
label_encoder = LabelEncoder()
train_embedding[target_col] = label_encoder.fit_transform(train_embedding[target_col])
test_embedding[target_col] = label_encoder.transform(test_embedding[target_col])

# Create a CVData object
cv_data = CVData(train_data=train_embedding, test_data=test_embedding, folds=5)
best_model = ModelCycle(cv_data=cv_data, target_col='target').get_best_model()

# train the best model
best_model.model.fit(train_embedding.drop(columns=[target_col]), train_embedding[target_col])

analyze_model(best_model.model, cv_data, target_label='target')

# # # # # Organize results
organized_results = pd.DataFrame(best_model.results)

# # # #
analyze_results(organized_results, best_model.parameters)
analyze_model(best_model.model, cv_data, target_label='target')
