import pandas as pd
from sklearn.model_selection import train_test_split

from chester.features_engineering.bag_of_words import get_bow_embedding
from chester.features_engineering.corex import get_corex_embedding
from chester.features_engineering.tfidf import get_tfidf_embedding
from chester.util import ReportCollector, REPORT_PATH


def get_embeddings(training_data: pd.DataFrame,
                   test_data: pd.DataFrame = None,
                   split_data: bool = True, split_prop: float = 0.3,
                   split_random_state=42,
                   text_column="clean_text", target_column='target',
                   corex=True, corex_dim=50, anchor_words=None, anchor_strength=1.6,
                   tfidf=True, tfidf_dim=100, bow=True, bow_dim=100,
                   ngram_range=(1, 2)):
    rc = ReportCollector(REPORT_PATH)
    rc.save_text("Extracting embedding")
    if split_data:
        if not test_data:
            training_data, test_data = train_test_split(training_data, test_size=split_prop,
                                                        random_state=split_random_state)

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
        title_to_print = f"{order_list[i]}, Extracting Corex topic model embeddings with dimension {corex_dim}"
        print(title_to_print)
        rc.save_text(title_to_print)
        corex_embedding, corex_test_embedding = \
            get_corex_embedding(training_data=training_data, test_data=test_data,
                                ngram_range=ngram_range, n_topics=corex_dim, text_column=text_column,
                                anchor_words=anchor_words, anchor_strength=anchor_strength)
        i += 1

    # Extract TF-IDF embeddings if requested
    if tfidf:
        title_to_print = f"{order_list[i]}, Extracting TF-IDF embeddings with dimension {tfidf_dim}"
        print(title_to_print)
        rc.save_text(title_to_print)
        tfidf_embedding, tfidf_test_embedding, _ = get_tfidf_embedding(training_data, test_df=test_data,
                                                                       ngram_range=ngram_range,
                                                                       embedding_size=tfidf_dim,
                                                                       text_column=text_column)
        i += 1

    # Extract bag-of-words embeddings if requested
    if bow:
        title_to_print = f"{order_list[i]}, Extracting bag-of-words embeddings with dimension {bow_dim}"
        print(title_to_print)
        rc.save_text(title_to_print)
        bow_embedding, bow_test_embedding, _ = get_bow_embedding(training_data, test_data=test_data,
                                                                 ngram_range=ngram_range,
                                                                 embedding_size=bow_dim,
                                                                 text_column=text_column)
        i += 1

    # Concatenate the embeddings and return them
    embeddings = pd.concat([corex_embedding, tfidf_embedding, ner_bow_embedding, bow_embedding], axis=1)
    test_embeddings = pd.concat(
        [corex_test_embedding, tfidf_test_embedding, ner_bow_test_embedding, bow_test_embedding], axis=1)

    title_to_print = f"Lastly, All embeddings have been concatenated"
    print(title_to_print)
    rc.save_text(title_to_print)
    # adding the label to train and test embedding
    try:
        training_data.reset_index(drop=True, inplace=True)
        embeddings.reset_index(drop=True, inplace=True)
        embeddings = pd.concat([embeddings, training_data[target_column]], axis=1)

        test_data.reset_index(drop=True, inplace=True)
        test_embeddings.reset_index(drop=True, inplace=True)
        test_embeddings = pd.concat([test_embeddings, test_data[target_column]], axis=1)
    except:
        pass
    return embeddings, test_embeddings


class TextFeatureExtraction:
    def __init__(self, training_data: pd.DataFrame = None,
                 test_data: pd.DataFrame = None,
                 split_data: bool = True, split_prop: float = 0.3, split_random_state=42,
                 text_column="text", target_column='target',
                 corex=True, corex_dim=50, tfidf=True, tfidf_dim=100, bow=True, bow_dim=100,
                 ngram_range=(1, 1)):
        self.training_data = training_data
        self.test_data = test_data
        self.split_data = split_data
        self.split_prop = split_prop
        self.split_random_state = split_random_state
        self.text_column = text_column
        self.target_column = target_column
        self.corex = corex
        self.corex_dim = corex_dim
        self.tfidf = tfidf
        self.tfidf_dim = tfidf_dim
        self.bow = bow
        self.bow_dim = bow_dim
        self.ngram_range = ngram_range


def extract_features(feature_extractor: TextFeatureExtraction):
    training_data = feature_extractor.training_data
    test_data = feature_extractor.test_data
    split_data = feature_extractor.split_data
    split_prop = feature_extractor.split_prop
    split_random_state = feature_extractor.split_random_state
    text_column = feature_extractor.text_column
    target_col = feature_extractor.target_column
    corex = feature_extractor.corex
    corex_dim = feature_extractor.corex_dim
    tfidf = feature_extractor.tfidf
    tfidf_dim = feature_extractor.tfidf_dim
    bow = feature_extractor.bow
    bow_dim = feature_extractor.bow_dim
    ngram_range = feature_extractor.ngram_range

    return get_embeddings(training_data, test_data, split_data, split_prop, split_random_state, text_column, target_col,
                          corex, corex_dim, tfidf, tfidf_dim, bow, bow_dim, ngram_range)
