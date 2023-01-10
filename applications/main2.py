import nltk
import pandas as pd
from nltk.corpus import brown

from cleaning.cleaning import *
from data_loader.webtext_data import get_chat_logs
from preprocessing.preprocessing import preprocess_text
from pretrained_models.sentiment import *
from text_analyzer import common_words
from text_analyzer.corex_topics import get_top_words, plot_corex_wordcloud
from text_analyzer.data_quality import analyze_text_data
from text_analyzer.word_cloud import create_word_cloud
from util import get_stopwords
import nltk

# nltk.download('webtext')
from nltk.corpus import webtext

# fileids = webtext.fileids()
# print(fileids)
if __name__ == '__main__':
    # access the data
    df = get_chat_logs()


    # Clean the text column
    df['text'] = df['text'].apply(lambda x: clean_text(x,
                                                       remove_stopwords_flag=True,
                                                       stopwords=get_stopwords()))

    # preprocess the text column
    df['clean_text'] = df['text'].apply(lambda x: preprocess_text(x, stem_flag=False))


    # basic stats
    analyze_text_data(df)

    # sentiment
    df = analyze_sentiment(df)
    print(report_sentiment_stats(df))
    plot_sentiment_scores(df)

    most_common_words = common_words.most_common_words(df, n=10)
    print(most_common_words)
    #
    # # word cloud
    create_word_cloud(df)

    # corex topic modeling
    # Example usage
    top_words_list = get_top_words(df, 5, 4, ngram_range=(1, 3))
    plot_corex_wordcloud(df)





    # brown_sent = brown.sents(categories=['reviews'])[:100]
    # brown_sent = [' '.join(x) for x in brown_sent]
    # df = pd.DataFrame({'text': brown_sent})
    #
    # # Clean the text column
    # df['text'] = df['text'].apply(lambda x: clean_text(x,
    #                                                    remove_stopwords_flag=True,
    #                                                    stopwords=get_stopwords()))
    #
    # # preprocess the text column
    # df['clean_text'] = df['text'].apply(lambda x: preprocess_text(x, stem_flag=False))

    # basic stats
    # analyze_text_data(df)

    # use the get_most_common_words function to get the list of most common words
    # most_common_words = common_words.most_common_words(df, n=10)

    # word cloud
    # create_word_cloud(df)

    # corex
    # plot_corex_wordcloud(df)

    # print(extractive_summarization(df))
