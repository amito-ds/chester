import pandas as pd

from tcap.text_analyzer import common_words as cw, corex_topics as corex, keywords_extraction as kw_extract

from tcap.text_analyzer.data_quality import analyze_text_stats, TextAnalyzer
from tcap.text_analyzer.key_sentences import extract_key_sentences
from tcap.text_analyzer.sentiment import analyze_sentiment, report_sentiment_stats, plot_sentiment_scores
from tcap.text_analyzer.word_cloud import create_word_cloud

data_quality_message = "Before we start analyzing the text, it's important to make sure that the data we are working " \
                       "with is clean and of good quality. The following report provides some key statistics about the " \
                       "data, such as the number of rows with missing data, " \
                       "number of unique words, average and median type-token ratio, average and median number " \
                       "of words and sentences per text, as well as average and median length of text.\n"

common_words_message = "One of the most basic text analysis techniques is counting the frequency of words in the text. " \
                       "The following report shows the most common words in the text data, which can give us an idea of" \
                       " the overall topic and content of the text.\n"

word_cloud_message = "A word cloud is a visual representation of the most common words in a piece of text, where the " \
                     "size of each word corresponds to its frequency. The word cloud can help us quickly identify " \
                     "the main themes and topics in the text data.\n"

sentiment_analysis_message = "Sentiment analysis is a technique used to determine the emotional tone of a piece of " \
                             "text. The following report shows the sentiment of the text data and provides " \
                             "a breakdown of positive, negative, and neutral sentiments.\n"

corex_topic_message = "Corex is a topic modeling technique that helps identify latent topics in the text data. " \
                      "The following report shows the top topics extracted from the text data and provides a " \
                      "word cloud for each topic.\n"

key_sentences_message = "Extracting key sentences from a piece of text is a technique used to identify the most " \
                        "important or representative sentences in the text. The following report shows the top" \
                        " sentences for each topic of the text data.\n"

kws_message = "Extracting key words from a piece of text is a technique used to identify the most important " \
              "or representative words in the text. " \
              "The following report shows the top words for each topic of the text data.\n"

ner_message = "\nExtracting named entities from a piece of text is a technique used to identify and categorize proper " \
              "nouns and terms that represent people, organizations, locations and more. " \
              "The following report shows the top named entities for each category of the text data.\n"


def print_analyze_message(create_wordcloud: bool = True,
                          corex_topics: bool = True,
                          key_sentences: bool = True,
                          common_words: bool = False,
                          sentiment: bool = False,
                          data_quality: bool = False,
                          corex_topics_num: int = 10,
                          top_words: int = 10,
                          n_sentences: int = 5):
    order_of_operations = ["First", "Next", "Then", "Additionally", "Furthermore", "Finally"]
    operations_counter = 0
    if data_quality:
        print(
            f"{order_of_operations[operations_counter]} we will analyze text statistics such as the number of words, unique words and type-token ratio.")
        operations_counter += 1
    if create_wordcloud:
        print(
            f"{order_of_operations[operations_counter]} we will create a wordcloud to visualize the most common words in the text.")
        operations_counter += 1
    if corex_topics:
        print(
            f"{order_of_operations[operations_counter]} we will extract {corex_topics_num} key topics using Corex and present them with their top words.")
        operations_counter += 1
    if key_sentences:
        print(f"{order_of_operations[operations_counter]} we will identify key sentences in the text.")
        operations_counter += 1
    if common_words:
        print(
            f"{order_of_operations[operations_counter]} we will extract the {top_words} most common words from the text.")
        operations_counter += 1
    if sentiment:
        print(f"{order_of_operations[operations_counter]} we will analyze the sentiment of the text.")
        operations_counter += 1
    print("Finally, Text analysis completed")


def analyze_text(df: pd.DataFrame,
                 create_wordcloud: bool = True,
                 corex_topics: bool = True,
                 key_sentences: bool = True,
                 common_words: bool = True,
                 sentiment: bool = True,
                 data_quality: bool = True,
                 kewords_extraction: bool = True,
                 corex_topics_num: int = 10,
                 top_words: int = 10,
                 n_sentences: int = 5,
                 text_column: str = 'text'):
    """
    Analyze text using various text analysis techniques.
    :param df: pandas data with clean text column
    :param create_wordcloud: flag indicating whether to create a word cloud (default: True)
    :param corex_topics: flag indicating whether to extract corex topics (default: True)
    :param key_sentences: flag indicating whether to extract key sentences (default: True)
    :param common_words: flag indicating whether to extract common words (default: True)
    :param sentiment: flag indicating whether to perform sentiment analysis (default: True)
    :param data_quality: flag indicating whether to check data quality (default: True)
    :param kewords_extraction: flag indicating whether to create potential keywords
    :param corex_topics_num: number of corex topics to extract (default: 10)
    :param top_words: top words
    :param n_sentences: number of sentences to return
    """

    print_analyze_message()
    if data_quality:
        print(data_quality_message)
        analyze_text_stats(df)
    if common_words:
        print(common_words_message)
        print(cw.most_common_words(df, common_words=top_words))
    if create_wordcloud:
        print(word_cloud_message)
        create_word_cloud(df)
    if sentiment:
        print(sentiment_analysis_message)
        df = df.copy()
        df = analyze_sentiment(df)
        print(report_sentiment_stats(df))
        plot_sentiment_scores(df)
    if corex_topics:
        print(corex_topic_message)
        corex.plot_corex_wordcloud(df, n_topics=corex_topics_num, top_words=top_words)
    if key_sentences:
        print(key_sentences_message)
        print(extract_key_sentences(df, n_sentences=n_sentences))
    if kewords_extraction:
        print(kws_message)
        full_text = '. '.join(df[text_column])
        kws_rake = kw_extract.RakeKeywordExtractor()
        print(kws_rake.extract(text=full_text))

def analyze_text_df(text_analyzer: TextAnalyzer):
    df = text_analyzer.df
    analyze_text(df,
                 create_wordcloud=text_analyzer.create_wordcloud,
                 corex_topics=text_analyzer.corex_topics,
                 key_sentences=text_analyzer.key_sentences,
                 common_words=text_analyzer.common_words,
                 sentiment=text_analyzer.sentiment,
                 data_quality=text_analyzer.data_quality,
                 corex_topics_num=text_analyzer.corex_topics_num,
                 top_words=text_analyzer.top_words,
                 n_sentences=text_analyzer.n_sentences)
