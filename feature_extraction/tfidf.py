from typing import Tuple

import pandas as pd
from nltk.corpus import brown
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import nltk
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

from cleaning import clean_text
from preprocessing import preprocess_text, get_stemmer
from util import get_stopwords


def get_tfidf_embedding(train_df: pd.DataFrame, text_column: str, test_df: pd.DataFrame = None,
                        ngram_range: Tuple[int, int] = (1, 1)):
    """
    Creates a TF-IDF embedding for the text data in the given DataFrames.
    Parameters:
    - train_df: A DataFrame with the training data.
    - test_df: A DataFrame with the test data.
    - text_column: The name of the column in the DataFrames that contains the text data.
    - ngram_range: The range of n-grams to consider when creating the embedding.

    Returns:
    - A tuple with the following elements:
        - A DataFrame with the TF-IDF embedding for the training data.
        - A DataFrame with the TF-IDF embedding for the test data (if provided).
        - The TfidfVectorizer object used to create the embedding.
    """

    # Create the TfidfVectorizer object
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=ngram_range)

    # Fit the vectorizer on the training data
    X_train = vectorizer.fit_transform(train_df[text_column])

    # Get the feature names
    feature_names = vectorizer.get_feature_names()

    # Create the embedding DataFrame for the training data
    embedding_train_df = pd.DataFrame(X_train.toarray(), columns=[f"tfidf_{word}" for word in feature_names])

    # Check if test data was provided
    if test_df is not None:
        # Transform the test data using the vectorizer
        X_test = vectorizer.transform(test_df[text_column])

        # Create the embedding DataFrame for the test data
        embedding_test_df = pd.DataFrame(X_test.toarray(), columns=[f"tfidf_{word}" for word in feature_names])

        return embedding_train_df, embedding_test_df, vectorizer
    else:
        return embedding_train_df, None, vectorizer


if __name__ == '__main__':
    brown_sent = brown.sents(categories='news')[:100]
    brown_sent = [' '.join(x) for x in brown_sent]
    df = pd.DataFrame({'text': brown_sent})

    # Clean the text column
    df['text'] = df['text'].apply(lambda x: clean_text(x,
                                                       remove_stopwords_flag=True,
                                                       stopwords=get_stopwords()))

    # preprocess the text column
    df['clean_text'] = df['text'].apply(lambda x: preprocess_text(x, stemmer=get_stemmer('porter'), stem_flag=True))

    # Extract the bag of words embedding
    bow_embedding, _, _ = get_tfidf_embedding(train_df=df, text_column='clean_text', ngram_range=(1, 2))

    print(bow_embedding[0:10])
