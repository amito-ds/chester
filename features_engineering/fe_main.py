import pandas as pd

from features_engineering.bag_of_words import get_bow_embedding
from features_engineering.corex import get_corex_embedding
from features_engineering.tfidf import get_tfidf_embedding


def get_embeddings(training_data: pd.DataFrame, test_data: pd.DataFrame = None, text_column="clean_text", corex=True,
                   corex_dim=10, tfidf=True, tfidf_dim=10000,
                   bow=True, bow_dim=10000,
                   ngram_range=(1, 1)):
    order_list = ["First", "Next", "Then", "Additionally", "Furthermore", "Then", "Additionally", "Furthermore", "Then",
                  "Additionally"]
    # Define empty DataFrames for the embeddings
    corex_embedding = pd.DataFrame()
    tfidf_embedding = pd.DataFrame()
    bow_embedding = pd.DataFrame()
    ner_bow_embedding = pd.DataFrame()
    corex_test_embedding = pd.DataFrame()
    tfidf_test_embedding = pd.DataFrame()
    bow_test_embedding = pd.DataFrame()
    ner_bow_test_embedding = pd.DataFrame()
    i = 0
    # Extract Corex topic model embeddings if requested
    if corex:
        print(f"{order_list[i]}, Extracting Corex topic model embeddings with dimension {corex_dim}")
        corex_embedding, corex_test_embedding = get_corex_embedding(training_data=training_data, test_data=test_data,
                                                                    ngram_range=ngram_range,
                                                                    n_topics=corex_dim, text_column=text_column)
        i += 1

    # Extract TF-IDF embeddings if requested
    if tfidf:
        print(f"{order_list[i]}, Extracting TF-IDF embeddings with dimension {tfidf_dim}")
        tfidf_embedding, tfidf_test_embedding, _ = get_tfidf_embedding(training_data, test_df=test_data,
                                                                       ngram_range=ngram_range,
                                                                       embedding_size=tfidf_dim,
                                                                       text_column=text_column)
        i += 1

    # Extract bag-of-words embeddings if requested
    if bow:
        print(f"{order_list[i]}, Extracting bag-of-words embeddings with dimension {bow_dim}")
        bow_embedding, bow_test_embedding, _ = get_bow_embedding(training_data, test_data=test_data,
                                                                 ngram_range=ngram_range,
                                                                 embedding_size=bow_dim,
                                                                 text_column=text_column)
        i += 1

    # Concatenate the embeddings and return them
    embeddings = pd.concat([corex_embedding, tfidf_embedding, ner_bow_embedding, bow_embedding], axis=1)
    test_embeddings = pd.concat(
        [corex_test_embedding, tfidf_test_embedding, ner_bow_test_embedding, bow_test_embedding], axis=1)
    print(f"Lastly, All embeddings have been concatenated")
    return embeddings, test_embeddings

# def get_embeddings(df, text_column="clean_text", corex=True, corex_dim=10, tfidf=True, tfidf_dim=10000,
#                    bow=True, bow_dim=10000,
#                    ngram_range=(1, 1)):
#     # Define empty DataFrames for the embeddings
#     corex_embedding = pd.DataFrame()
#     tfidf_embedding = pd.DataFrame()
#     bow_embedding = pd.DataFrame()
#     ner_bow_embedding = pd.DataFrame()
#
#     # Extract Corex topic model embeddings if requested
#     if corex:
#         corex_embedding = get_corex_embedding(df, ngram_range=ngram_range, n_topics=corex_dim, text_column=text_column)
#
#     # Extract TF-IDF embeddings if requested
#     if tfidf:
#         tfidf_embedding, _, _ = get_tfidf_embedding(df, ngram_range=ngram_range, embedding_size=tfidf_dim,
#                                                     text_column=text_column)
#
#     # Extract bag-of-words embeddings if requested
#     if bow:
#         bow_embedding, _, _ = get_bow_embedding(df, ngram_range=ngram_range, embedding_size=bow_dim,
#                                                 text_column=text_column)
#
#     # Concatenate the embeddings and return them
#     embeddings = pd.concat([corex_embedding, tfidf_embedding, ner_bow_embedding, bow_embedding], axis=1)
#     return embeddings
