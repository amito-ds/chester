import numpy as np
from nltk import WordNetLemmatizer
from nltk.corpus import brown

from cleaning import *
import nltk

# Load the list of stopwords

import nltk

from data_quality import confidence_interval
from preprocessing import lemmatize, get_stemmer, preprocess_text

import pandas as pd
from text_analyzer import common_words
from text_analyzer.text_summary import extractive_summarization
from text_analyzer.word_cloud import create_word_cloud
from util import get_stopwords
import random

# nltk.download('brown')


if __name__ == '__main__':
    # # define a list of words to use in the reviews
    # words = ['clean', 'comfortable', 'friendly', 'nice', 'great', 'excellent', 'amazing', 'fantastic', 'lovely',
    #          'wonderful', 'perfect', 'enjoyable', 'delightful', 'beautiful', 'gorgeous']
    #
    # # generate a list of 30 reviews with up to 10 words each
    # reviews = [' '.join([random.choice(words) for _ in range(random.randint(1, 11))]) for _ in range(30)]

    # ['adventure', 'belles_lettres', 'fiction', 'humor', 'lore', 'mystery', 'news', 'romance', 'science_fiction']
    brown_sent = brown.sents(categories='news')[:100]
    brown_sent = [' '.join(x) for x in brown_sent]
    df = pd.DataFrame({'text': brown_sent})

    # Clean the text column
    df['text'] = df['text'].apply(lambda x: clean_text(x,
                                                       remove_stopwords_flag=True,
                                                       stopwords=get_stopwords()))

    # preprocess the text column
    df['text'] = df['text'].apply(lambda x: preprocess_text(x, stem_flag=False))
    # print(calculate_text_statistics(df))


    # df['text'] = df['text'].apply(lambda x: preprocess_text(x, stem_flag=False, stemmer=get_stemmer("snowball")))

    # use the get_most_common_words function to get the list of most common words
    # most_common_words = common_words.most_common_words(df, n=10)
    # print(most_common_words)

    # word cloud
    # create_word_cloud(df)
    # print(extractive_summarization(df))
