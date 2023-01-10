import pandas as pd
import spacy
from nltk.corpus import brown

from cleaning import clean_text
from preprocessing import preprocess_text
from util import get_stopwords
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as ss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize

from corextopic import corextopic as ct


def train_corex_model(training_data, test_data=None, text_column='clean_text', ngram_range=(1, 1), n_topics=10,
                      max_features=10000):
    # Preprocess data
    vectorizer = CountVectorizer(stop_words='english', max_features=max_features, binary=True, ngram_range=ngram_range)
    doc_word = vectorizer.fit_transform(training_data[text_column])
    doc_word = ss.csr_matrix(doc_word)
    feature_names = list(vectorizer.vocabulary_.keys())
    words = list(np.asarray(feature_names))

    # Train model
    topic_model = ct.Corex(n_hidden=n_topics, words=words, max_iter=200, verbose=False, seed=1)
    topic_model.fit(doc_word, words=words)

    # Get the topic probabilities for the training data
    topic_probs = topic_model.transform(doc_word, details=True)[0]
    # Normalize the topic probabilities
    topic_probs /= topic_probs.sum(axis=1)[:, np.newaxis]
    # Create a DataFrame of topic probability features
    topic_prob_df = pd.DataFrame(topic_probs, columns=[f"corex_topic_{i + 1}" for i in range(n_topics)])

    if test_data is not None:
        # Preprocess test data
        test_doc_word = vectorizer.transform(test_data[text_column])
        test_doc_word = ss.csr_matrix(test_doc_word)
        # Get the topic probabilities for the test data
        test_topic_probs = topic_model.transform(test_doc_word, details=True)[0]
        # Normalize the topic probabilities
        test_topic_probs /= test_topic_probs.sum(axis=1)[:, np.newaxis]
        # Create a DataFrame of topic probability features for the test data
        test_topic_prob_df = pd.DataFrame(test_topic_probs, columns=[f"corex_topic_{i + 1}" for i in range(n_topics)])

        return topic_prob_df, test_topic_prob_df
    else:
        return topic_prob_df


if __name__ == '__main__':
    brown_sent = brown.sents(categories=['reviews', 'news'])[:1000]
    brown_sent = [' '.join(x) for x in brown_sent]
    df = pd.DataFrame({'text': brown_sent})

    # Clean the text column
    df['text'] = df['text'].apply(lambda x: clean_text(x,
                                                       remove_stopwords_flag=True,
                                                       stopwords=get_stopwords()))

    # preprocess the text column
    df['clean_text'] = df['text'].apply(lambda x: preprocess_text(x, stem_flag=False))

    # corex
    corex_embedding = train_corex_model(df)
    print(np.sum(corex_embedding , axis=1))
