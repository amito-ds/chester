# THIS EXAMPLE WORKS SO DONT CHANGE THE CODE!!!!


import pandas as pd
from sklearn.preprocessing import LabelEncoder

from cleaning.cleaning_func import clean_text
from data_loader.webtext_data import load_data_pirates, load_data_king_arthur
from tcap.feature_analyzing import PreModelAnalysis
from features_engineering.feature_main import get_embeddings
from preprocessing.preprocessing_func import preprocess_text, get_stemmer
from tcap.util import get_stopwords

df1 = load_data_pirates().assign(target='chat_logs')
df2 = load_data_king_arthur().assign(target='pirates')
df = pd.concat([df1, df2])


#
# # # Clean the text column
get_sw = get_stopwords()
df['text'] = df['text'].apply(lambda x: clean_text(x,
                                                   remove_stopwords_flag=True,
                                                   stopwords=get_sw))
#
# # preprocess the text column
df['text'] = df['text'].apply(lambda x:
                                    preprocess_text(x, stemmer=get_stemmer('porter'), stem_flag=True))
#
# #
train_embedding, test_embedding = get_embeddings(
    training_data=df,
    corex=True, corex_dim=100,
    tfidf=True, tfidf_dim=100,
    bow=True, bow_dim=100)

target_col = 'target'
label_encoder = LabelEncoder()
train_embedding[target_col] = label_encoder.fit_transform(train_embedding[target_col])
test_embedding[target_col] = label_encoder.transform(test_embedding[target_col])


pma = PreModelAnalysis(train_embedding, target_column=target_col)
pma.run()
