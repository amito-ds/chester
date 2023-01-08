import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as ss
from corextopic import corextopic as ct

from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def plot_wordcloud(df, top_words=10, n_topics=5):
    # Get top words and weights
    top_words_list = get_top_words(df, top_words, n_topics)

    # Combine words and weights into a single list
    words = [word for word, weight in top_words_list]
    weights = [weight for word, weight in top_words_list]

    # Create word cloud
    wordcloud = WordCloud(width=800, height=400)
    wordcloud.generate_from_frequencies(dict(zip(words, weights)))

    # Plot word cloud
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()



def get_top_words(df, top_words, n_topics):
    # Preprocess data
    vectorizer = CountVectorizer(stop_words='english', max_features=20000, binary=True)
    doc_word = vectorizer.fit_transform(df['text'])
    doc_word = ss.csr_matrix(doc_word)
    words = list(np.asarray(vectorizer.get_feature_names()))

    # Train model
    topic_model = ct.Corex(n_hidden=n_topics, words=words, max_iter=200, verbose=False, seed=1)
    topic_model.fit(doc_word, words=words)

    # Get top words and weights for each topic
    topics = topic_model.get_topics()
    if topics is None:
        return []

    top_words_list = []
    for i, topic in enumerate(topics):
        topic_words, weights, _ = zip(*topic)
        num_words = min(top_words, len(topic_words))  # Use smaller of n and num words in topic
        top_words_list += [(topic_words[j], weights[j]) for j in range(num_words)]
        print('{}: '.format(i) + ', '.join(topic_words))

    return top_words_list
