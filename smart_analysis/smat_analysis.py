import numpy as np

from cleaning.cleaning import *
from data_loader.webtext_data import *
from features_engineering.fe_main import get_embeddings
from preprocessing.preprocessing import preprocess_text
from smart_analysis.similar import get_most_similar_texts
from util import get_stopwords

if __name__ == '__main__':
    df = load_data_chat_logs()

    # Clean the text column
    df['text'] = df['text'].apply(lambda x: clean_text(x,
                                                       remove_stopwords_flag=True,
                                                       stopwords=get_stopwords()))

    # preprocess the text column
    df['clean_text'] = df['text'].apply(lambda x: preprocess_text(x, stem_flag=False))

    # basic stats
    # analyze_text(df, common_words=True, sentiment=True, data_quality=True)

    # get embedding
    embedding = get_embeddings(df, corex_dim=2)
    # print(embedding.columns)

    df_embedding = pd.concat([df['clean_text'], embedding], axis=1)

    print(get_most_similar_texts(df_embedding, index=10, top_n=10))
