from collections import Counter

import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from chester.util import ReportCollector, REPORT_PATH


def extract_key_sentences_lsa(text, k=10):
    """
    Extract k most important sentences from text using LSA algorithm.
    :param text: the text to extract key sentences from
    :param k: the number of sentences to extract (default: 10)
    :return: list of k most important sentences
    """
    # tokenize the text into sentences
    sentences = sent_tokenize(text)

    # apply tf-idf weighting to the sentences
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)

    # apply LSA to the tf-idf weighted sentences
    svd = TruncatedSVD(n_components=k)
    X = svd.fit_transform(X)

    # use cosine similarity to identify the most important sentences
    similarity = np.asarray(np.asmatrix(X) * np.asmatrix(X).T)
    sentence_scores = similarity.diagonal()

    # sort the sentences by importance
    sentence_scores_sorted = sentence_scores.argsort()[::-1]
    return_list = [sentences[i] for i in sentence_scores_sorted]
    return return_list


def key_sentences(df, text_column='text', common_sentences=10):
    # Concatenate all text in the specified column into one string
    all_text = ". ".join(df[text_column])

    # Tokenize the string into sentences
    sentences = sent_tokenize(all_text)

    # Create a Counter to store the frequency of each sentence
    cnt = Counter()
    for sentence in sentences:
        cnt[sentence] += 1

    # return the most common sentences
    return cnt.most_common(common_sentences)


def extract_key_sentences(df: pd.DataFrame, text_column='text', algorithm='LSA', n_sentences=10):
    """
    Extract k most important sentences from a pandas dataframe containing text data.
    :param df: the dataframe to extract key sentences from
    :param text_column: the name of the column containing the text data (default: 'clean_text')
    :param algorithm: the algorithm to use for sentence extraction (default: 'summarization')
    :param n_sentences: the number of sentences to extract (default: 10)
    :return: list of k most important sentences
    """
    rc = ReportCollector(REPORT_PATH)

    full_text = '. '.join(df[text_column])
    if algorithm == 'LSA':
        sentences = list(set(extract_key_sentences_lsa(full_text, n_sentences)))[0:n_sentences]
    elif algorithm == 'common_sentences':
        tokenized_sentences = [sentence.split() for sentence in sent_tokenize(full_text)]
        sentence_counter = Counter([tuple(sentence) for sentence in tokenized_sentences])
        common_sentences = [" ".join(sentence) for sentence, count in sentence_counter.most_common(n_sentences)]
        sentences = common_sentences
    else:
        raise ValueError(f'Invalid algorithm: {algorithm}')

    title = f'\tKey sentences out of based on {algorithm} algorithm: {sentences} \n'
    rc.save_text(title)
    return title


def extract_first_k_words(text, k):
    """
    Extract the first k words from text.
    :param text: the text to extract words from
    :param k: the number of words to extract
    :return: list of k first words in the text
    """
    words = text.split()  # split text into words using the whitespace as a delimiter
    return ' '.join(words[:k])  # return the first k words
