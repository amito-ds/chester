import numpy as np
import pandas as pd
import scipy.sparse as ss
from corextopic import corextopic as ct
from sklearn.feature_extraction.text import CountVectorizer


def get_corex_embedding(training_data, test_data=None, text_column='text', ngram_range=(1, 1),
                        n_topics=50,
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
    topic_probs = topic_probs / np.sum(topic_probs, axis=1, keepdims=True)
    # Create a DataFrame of topic probability features
    topic_prob_df = pd.DataFrame(topic_probs, columns=[f"corex_topic_{i + 1}" for i in range(n_topics)])

    if test_data is not None:
        # Preprocess test data
        test_doc_word = vectorizer.transform(test_data[text_column])
        test_doc_word = ss.csr_matrix(test_doc_word)
        # Get the topic probabilities for the test data
        test_topic_probs = topic_model.transform(test_doc_word, details=True)[0]
        # Normalize the topic probabilities
        test_topic_probs = test_topic_probs / np.sum(test_topic_probs, axis=1, keepdims=True)
        # Create a DataFrame of topic probability features for the test data
        test_topic_prob_df = pd.DataFrame(test_topic_probs, columns=[f"corex_topic_{i + 1}" for i in range(n_topics)])

        return topic_prob_df, test_topic_prob_df
    else:
        return topic_prob_df, None
