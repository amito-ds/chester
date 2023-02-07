import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as ss
from corextopic import corextopic as ct
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud


def plot_corex_wordcloud(df, top_words=10, n_topics=10, plot=False, text_column='text'):
    # Get top words and weights
    top_words_list = get_top_words(df, top_words, n_topics, text_column=text_column)

    if plot:
        fig, axs = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle("Top Words Word Clouds")

        for i, (words, weights) in enumerate(top_words_list[:9]):
            # Combine words and weights into a single list
            words = [word for word, weight in zip(words, weights)]
            weights = [weight for word, weight in zip(words, weights)]

            # Create word cloud
            wordcloud = WordCloud(width=800, height=400)
            wordcloud.generate_from_frequencies(dict(zip(words, weights)))

            # Plot word cloud
            axs[i // 3, i % 3].imshow(wordcloud, interpolation="bilinear")
            axs[i // 3, i % 3].axis("off")
            axs[i // 3, i % 3].set_title(f"Topic {i + 1}")

        plt.show()


def get_top_words(df: pd.DataFrame,
                  top_words,
                  n_topics,
                  max_features=1000,
                  text_column: str = 'text',
                  ngram_range=(1, 3)):
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
    for i, topic in enumerate(topics):
        topic_words, weights, _ = zip(*topic)
        num_words = min(top_words, len(topic_words))  # Use smaller of n and num words in topic
        top_words_list += [(topic_words[j], weights[j]) for j in range(num_words)]
        print('\tTopic {}: '.format(i + 1) + ', '.join(topic_words))
    print("\n")

    return top_words_list
