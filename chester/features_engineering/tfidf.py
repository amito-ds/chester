from typing import Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def get_tfidf_embedding(train_df: pd.DataFrame,
                        text_column: str = 'text',
                        test_df: pd.DataFrame = None,
                        ngram_range: Tuple[int, int] = (1, 1), embedding_size=100):
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
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=embedding_size)

    # Fit the vectorizer on the training data
    X_train = vectorizer.fit_transform(train_df[text_column])

    # Get the feature names
    feature_names = list(vectorizer.vocabulary_.keys())

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
