import pandas as pd

from features_engineering.bag_of_words import get_bow_embedding
from features_engineering.corex import get_corex_embedding
from features_engineering.tfidf import get_tfidf_embedding


def get_embeddings(df, text_column="clean_text", corex=True, corex_dim=10, tfidf=True, tfidf_dim=10000,
                   bow=True, bow_dim=10000,
                   ngram_range=(1, 1)):
    # Define empty DataFrames for the embeddings
    corex_embedding = pd.DataFrame()
    tfidf_embedding = pd.DataFrame()
    bow_embedding = pd.DataFrame()
    ner_bow_embedding = pd.DataFrame()

    # Extract Corex topic model embeddings if requested
    if corex:
        corex_embedding = get_corex_embedding(df, ngram_range=ngram_range, n_topics=corex_dim, text_column="clean_text")

    # Extract TF-IDF embeddings if requested
    if tfidf:
        tfidf_embedding, _, _ = get_tfidf_embedding(df, ngram_range=ngram_range, embedding_size=tfidf_dim,
                                                    text_column="clean_text")

    # Extract bag-of-words embeddings if requested
    if bow:
        bow_embedding, _, _ = get_bow_embedding(df, ngram_range=ngram_range, embedding_size=bow_dim,
                                                text_column="clean_text")

    # Concatenate the embeddings and return them
    embeddings = pd.concat([corex_embedding, tfidf_embedding, ner_bow_embedding, bow_embedding], axis=1)
    return embeddings
