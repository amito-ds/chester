import nltk
import pandas as pd
from nltk.corpus import brown

from cleaning.cleaning import *
from data_loader.webtext_data import *
from preprocessing.preprocessing import preprocess_text
from text_analyzer.sentiment import *
from text_analyzer import common_words
from text_analyzer.corex_topics import get_top_words, plot_corex_wordcloud
from text_analyzer.data_quality import analyze_text_stats
from text_analyzer.smart_text_analyzer import analyze_text
from text_analyzer.word_cloud import create_word_cloud
from util import get_stopwords
import nltk

# nltk.download('webtext')
from nltk.corpus import webtext

# fileids = webtext.fileids()
# print(fileids)
if __name__ == '__main__':
    # access the data
    # df = load_data_brown('news')
    df = load_data_chat_logs()

    # Clean the text column
    df['text'] = df['text'].apply(lambda x: clean_text(x,
                                                       remove_stopwords_flag=True,
                                                       stopwords=get_stopwords()))

    # preprocess the text column
    df['clean_text'] = df['text'].apply(lambda x: preprocess_text(x, stem_flag=False))

    # basic stats
    analyze_text(df, common_words=True, sentiment=True, data_quality=True)
