from nltk import WordNetLemmatizer

from cleaning import *
import nltk

# Load the list of stopwords

import nltk

from preprocessing import lemmatize, get_stemmer, preprocess_text

import pandas as pd
from text_analyzer import common_words
from text_analyzer.text_summary import extractive_summarization
from text_analyzer.word_cloud import create_word_cloud
from util import get_stopwords
import random

if __name__ == '__main__':
    # define a list of words to use in the reviews
    words = ['clean', 'comfortable', 'friendly', 'nice', 'great', 'excellent', 'amazing', 'fantastic', 'lovely',
             'wonderful', 'perfect', 'enjoyable', 'delightful', 'beautiful', 'gorgeous']

    # generate a list of 30 reviews with up to 10 words each
    reviews = [' '.join([random.choice(words) for _ in range(random.randint(1, 11))]) for _ in range(30)]
    df = pd.DataFrame({'text': reviews})

    # Clean the text column
    df['text'] = df['text'].apply(lambda x: clean_text(x,
                                                       remove_stopwords_flag=True,
                                                       stopwords=get_stopwords()))

    # use the get_most_common_words function to get the list of most common words
    # most_common_words = common_words.most_common_words(df, n=4)
    # print(most_common_words)

    # create_word_cloud(df)
    print(extractive_summarization(df))
