import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as ss
from corextopic import corextopic as ct
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

from chester.util import ReportCollector, REPORT_PATH


def plot_corex_wordcloud(df, top_words=20, n_topics=10, plot=True, text_column='text'):
    top_words_list = get_top_words(df, top_words, n_topics, text_column=text_column)
    top_words_list = pd.DataFrame(top_words_list, columns=["topic_index", "term", "weight"])
    if plot:
        # Plot the word clouds
        N = math.floor(math.sqrt(n_topics))
        fig, axs = plt.subplots(N, N, figsize=(N * 4, N * 4), dpi=100)
        fig.suptitle("Word Clouds for Top Word for Corex Topics", fontsize=16)

        for i in range(n_topics - 1):
            topic_words = top_words_list[top_words_list["topic_index"] == i]
            topic_words = dict(zip(topic_words["term"], topic_words["weight"]))
            if len(topic_words) == 0:
                break
            wordcloud = WordCloud(width=800, height=800, background_color='black',
                                  stopwords=None, min_font_size=10).generate_from_frequencies(topic_words)

            subplot_index = i + 1
            try:
                ax = plt.subplot(N, N, subplot_index)
                plt.imshow(wordcloud)
                plt.axis("off")
                plt.title("Topic {}".format(i), fontsize=14)
            except:
                plt.close()
                pass
        plt.tight_layout()
        plt.show()
        plt.close()


def get_top_words(df: pd.DataFrame,
                  top_words,
                  n_topics,
                  max_features=1000,
                  text_column: str = 'text',
                  ngram_range=(1, 3)):
    rc = ReportCollector(REPORT_PATH)
    # Preprocess data
    vectorizer = CountVectorizer(stop_words='english',
                                 max_features=max_features,
                                 binary=True,
                                 ngram_range=ngram_range)
    doc_word = vectorizer.fit_transform(df[text_column])
    doc_word = ss.csr_matrix(doc_word)
    feature_names = list(vectorizer.vocabulary_.keys())
    words = list(np.asarray(feature_names))

    # Train model
    topic_model = ct.Corex(n_hidden=n_topics, words=words, max_iter=200, verbose=False, seed=1)
    topic_model.fit(doc_word, words=words)

    # Get top words and weights for each topic
    topics = topic_model.get_topics()
    if topics is None:
        return []

    top_words_list = []
    rc.save_text("\nCorex topic analysis:")
    for i, topic in enumerate(topics):
        if len(topic) == 0:
            break
        topic_words, weights, _ = zip(*topic)
        num_words = min(top_words, len(topic_words))  # Use smaller of n and num words in topic
        top_words_list += [(i, topic_words[j], weights[j]) for j in range(num_words)]
        title = '\tTopic {}: '.format(i + 1) + ', '.join(topic_words)
        print(title)
        rc.save_text(text=title)
    print("\n")

    return top_words_list
