from typing import Tuple

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize


def get_bow_embedding(train_df: pd.DataFrame, column: str, test_df: pd.DataFrame = None,
                      ngram_range: Tuple[int, int] = (1, 1)):
    """
    Extract bag of words embeddings for the text in the given column of the DataFrame.

    Parameters
    ----------
    train_df: pd.DataFrame
        DataFrame containing the training data.
    test_df: pd.DataFrame, optional
        DataFrame containing the test data. If not provided, the function will only return the embeddings for the training data.
    column: str
        Name of the column in the DataFrame containing the text data.
    ngram_range: Tuple[int, int], optional
        Range of n-grams to consider. The default is (1, 1), which considers only unigrams.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, 'Model']
        Tuple containing the bag of words embeddings for the train and test sets, and the trained CountVectorizer model.
    """
    # Initialize the CountVectorizer
    vectorizer = CountVectorizer(ngram_range=ngram_range)

    # Fit and transform the training data
    X_train = vectorizer.fit_transform(train_df[column])

    # Normalize the training data
    X_train = normalize(X_train)

    # Extract feature names
    feature_names = list(vectorizer.vocabulary_.keys())

    # Create a DataFrame for the training data
    train_embedding_df = pd.DataFrame(X_train.todense(), columns=[f"bow_{word}" for word in feature_names])

    if test_df is not None:
        # Transform the test data
        X_test = vectorizer.transform(test_df[column])

        # Normalize the test data
        X_test = normalize(X_test)

        # Create a DataFrame for the test data
        test_embedding_df = pd.DataFrame(X_test.todense(), columns=[f"bow_{word}" for word in feature_names])

        return train_embedding_df, test_embedding_df, vectorizer

    return train_embedding_df, None, vectorizer


from nltk.corpus import brown

from cleaning import clean_text
from preprocessing import preprocess_text
from util import get_stopwords

if __name__ == '__main__':
    brown_sent = brown.sents(categories='news')[:100]
    brown_sent = [' '.join(x) for x in brown_sent]
    df = pd.DataFrame({'text': brown_sent})

    # Clean the text column
    df['text'] = df['text'].apply(lambda x: clean_text(x,
                                                       remove_stopwords_flag=True,
                                                       stopwords=get_stopwords()))

    # preprocess the text column
    df['clean_text'] = df['text'].apply(lambda x: preprocess_text(x, stem_flag=False))

    # Extract the bag of words embedding
    bow_embedding, _, _ = get_bow_embedding(train_df=df, column='clean_text', ngram_range=(1, 2))

    print(bow_embedding[0:10])
