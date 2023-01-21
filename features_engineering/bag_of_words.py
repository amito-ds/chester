from typing import Tuple

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize


def get_bow_embedding(training_data: pd.DataFrame,
                      text_column: str = 'text',
                      test_data: pd.DataFrame = None,
                      ngram_range: Tuple[int, int] = (1, 1),
                      embedding_size=100):
    """
    Extract bag of words embeddings for the text in the given column of the DataFrame.

    Parameters
    ----------
    training_data: pd.DataFrame
        DataFrame containing the training data.
    test_data: pd.DataFrame, optional
        DataFrame containing the test data. If not provided, the function will only return the embeddings for the training data.
    text_column: str
        Name of the column in the DataFrame containing the text data.
    ngram_range: Tuple[int, int], optional
        Range of n-grams to consider. The default is (1, 1), which considers only unigrams.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, 'Model']
        Tuple containing the bag of words embeddings for the train and test sets, and the trained CountVectorizer model.
    """
    # Initialize the CountVectorizer
    vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=embedding_size)

    # Fit and transform the training data
    X_train = vectorizer.fit_transform(training_data[text_column])

    # Normalize the training data
    X_train = normalize(X_train)

    # Extract feature names
    feature_names = list(vectorizer.vocabulary_.keys())

    # Create a DataFrame for the training data
    train_embedding_df = pd.DataFrame(X_train.todense(), columns=[f"bow_{word}" for word in feature_names])

    if test_data is not None:
        # Transform the test data
        X_test = vectorizer.transform(test_data[text_column])

        # Normalize the test data
        X_test = normalize(X_test)

        # Create a DataFrame for the test data
        test_embedding_df = pd.DataFrame(X_test.todense(), columns=[f"bow_{word}" for word in feature_names])
        return train_embedding_df, test_embedding_df, vectorizer

    return train_embedding_df, None, vectorizer
