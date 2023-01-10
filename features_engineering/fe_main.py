import pandas as pd
import spacy

from features_engineering.corex import get_corex_embedding

from features_engineering.tfidf import get_tfidf_embedding

from features_engineering.bag_of_words import get_bow_embedding

from features_engineering.ner import get_ner_bow_embedding, classify_text
from nltk.corpus import brown

from cleaning import clean_text
from preprocessing import preprocess_text
from util import get_stopwords


def get_embeddings(text_data, text_column="clean_text", corex=False, corex_dim=10, tfidf=False, tfidf_dim=10000,
                   bow=False, bow_dim=10000,
                   ngram_range=(1, 1)):
    # Define empty DataFrames for the embeddings
    corex_embedding = pd.DataFrame()
    tfidf_embedding = pd.DataFrame()
    bow_embedding = pd.DataFrame()
    ner_bow_embedding = pd.DataFrame()

    # Extract Corex topic model embeddings if requested
    if corex:
        corex_embedding = get_corex_embedding(text_data, ngram_range=ngram_range, n_topics=corex_dim)

    # Extract TF-IDF embeddings if requested
    if tfidf:
        tfidf_embedding, _, _ = get_tfidf_embedding(text_data, ngram_range=ngram_range, embedding_size=tfidf_dim)

    # Extract bag-of-words embeddings if requested
    if bow:
        bow_embedding, _, _ = get_bow_embedding(text_data, ngram_range=ngram_range, embedding_size=bow_dim)
        # NER
        df["class"] = df[text_column].apply(classify_text)
        ner_bow_embedding = get_ner_bow_embedding(df)

    # Concatenate the embeddings and return them
    embeddings = pd.concat([corex_embedding, tfidf_embedding, ner_bow_embedding, bow_embedding], axis=1)
    return embeddings


if __name__ == '__main__':
    path = '/Users/amitosi/opt/anaconda3/envs/py39/lib/python3.9/site-packages/en_core_web_md/en_core_web_md-3.3.0'
    nlp = spacy.load(path)
    brown_sent = brown.sents(categories=['reviews'])[:100]
    brown_sent = [' '.join(x) for x in brown_sent]
    df = pd.DataFrame({'text': brown_sent})

    # Clean the text column
    df['text'] = df['text'].apply(lambda x: clean_text(x,
                                                       remove_stopwords_flag=True,
                                                       stopwords=get_stopwords()))

    # preprocess the text column
    df['clean_text'] = df['text'].apply(lambda x: preprocess_text(x, stem_flag=False))

    print(get_embeddings(df, corex=True, tfidf=True, bow=True, tfidf_dim=3, bow_dim=3)[0:10])
