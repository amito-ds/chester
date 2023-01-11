import pandas as pd

from text_analyzer.common_words import most_common_words
from text_analyzer.corex_topics import *
from text_analyzer.data_quality import analyze_text_stats
from text_analyzer.key_sentences import extract_key_sentences
from text_analyzer.sentiment import analyze_sentiment, report_sentiment_stats, plot_sentiment_scores
from text_analyzer.word_cloud import create_word_cloud


def analyze_text(df: pd.DataFrame,
                 create_wordcloud: bool = True,
                 corex_topics: bool = True,
                 key_sentences: bool = True,
                 common_words: bool = False,
                 sentiment: bool = False,
                 data_quality: bool = False,
                 corex_topics_num: int = 10,
                 top_words: int = 10,
                 n_sentences: int = 5):
    """
    Analyze text using various text analysis techniques.
    :param df: pandas data with clean text column
    :param create_wordcloud: flag indicating whether to create a word cloud (default: True)
    :param corex_topics: flag indicating whether to extract corex topics (default: True)
    :param key_sentences: flag indicating whether to extract key sentences (default: True)
    :param common_words: flag indicating whether to extract common words (default: False)
    :param sentiment: flag indicating whether to perform sentiment analysis (default: False)
    :param data_quality: flag indicating whether to check data quality (default: False)
    :param corex_topics_num: number of corex topics to extract (default: 10)
    :param top_words: top words
    :param n_sentences: number of sentences to return
    """
    if data_quality:
        analyze_text_stats(df)
    if common_words:
        most_common_words(df, common_words=top_words)
    if create_wordcloud:
        create_word_cloud(df)
    if sentiment:
        df = df.copy()
        df = analyze_sentiment(df)
        plot_sentiment_scores(df)
    print(report_sentiment_stats(df))
    if corex_topics:
        plot_corex_wordcloud(df, n_topics=corex_topics_num, top_words=top_words)
    if key_sentences:
        extract_key_sentences(df, n_sentences=n_sentences, top_words=top_words)
