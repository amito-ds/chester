import pandas as pd

from chester.run.user_classes import TextSummary
from chester.text_stats_analysis import common_words as cw, corex_topics as corex, keywords_extraction as kw_extract
from chester.text_stats_analysis.data_quality import analyze_text_stats, TextAnalyzer
from chester.text_stats_analysis.key_sentences import extract_key_sentences
from chester.text_stats_analysis.sentiment import analyze_sentiment, report_sentiment_stats, plot_sentiment_scores
from chester.text_stats_analysis.word_cloud import create_word_cloud
from chester.util import ReportCollector, REPORT_PATH

data_quality_message = "Before analyzing text, ensure data is clean and of good quality.\n" \
                       "Report provides key stats on data quality, incl. missing data,\n" \
                       "unique words, type-token ratio, words/sentences per text, and text length:"

common_words_message = "Count word frequency for basic text analysis.\n" \
                       "Report shows most common words, giving insight into text topic and content:"

word_cloud_message = "A word cloud visualizes most common words in text. Size indicates frequency.\n" \
                     "It quickly identifies text data's main themes and topics.\n"

sentiment_analysis_message = "\nSentiment analysis determines emotional tone of text.\n" \
                             "Report shows sentiment breakdown: positive, negative, neutral:"

corex_topic_message = "Corex topic modeling identifies latent topics in text data.\n" \
                      "Report shows top topics, each topic represented by the following words:"

key_sentences_message = "Extracting key sentences identifies important/representative sentences in text.\n" \
                        "Report shows top sentences for each text topic:\n"

kws_message = "Extracting key words identifies important/representative words in text.\n" \
              "Report shows top words for each text topic:"


def print_analyze_message(create_wordcloud: bool = True,
                          corex_topics: bool = True,
                          key_sentences: bool = True,
                          common_words: bool = True,
                          sentiment: bool = True,
                          data_quality: bool = True,
                          corex_topics_num: int = 10,
                          top_words: int = 10,
                          n_sentences: int = 5):
    order_of_operations = ["First,", "Next,", "Then,", "Additionally,", "Furthermore,", "Later,"]
    operations_counter = 0
    if data_quality:
        print(
            f"{order_of_operations[operations_counter]} "
            f"we will analyze text statistics such as the number of words, unique words and type-token ratio.")
        operations_counter += 1
    if create_wordcloud:
        print(
            f"{order_of_operations[operations_counter]} "
            f"we will create a wordcloud to visualize the most common words in the text.")
        operations_counter += 1
    if corex_topics:
        print(
            f"{order_of_operations[operations_counter]} "
            f"we will extract {corex_topics_num} key topics using Corex and present them with their top words.")
        operations_counter += 1
    if key_sentences:
        print(f"{order_of_operations[operations_counter]} "
              f"we will identify key sentences in the text.")
        operations_counter += 1
    if common_words:
        print(
            f"{order_of_operations[operations_counter]} "
            f"we will extract the {top_words} most common words from the text.")
        operations_counter += 1
    if sentiment:
        print(f"{order_of_operations[operations_counter]} "
              f"we will analyze the sentiment of the text.")
        operations_counter += 1
    print("The results of text cleaning and preprocessing, including key words and key sentences,\n"
          "may not accurately reflect the original content and could lead to incorrect conclusions,\n "
          "if not handled properly.\n")


def analyze_text(df: pd.DataFrame,
                 create_wordcloud: bool = True,
                 corex_topics: bool = True,
                 key_sentences: bool = True,
                 common_words: bool = True,
                 data_quality: bool = True,
                 kewords_extraction: bool = True,
                 corex_topics_num: int = 10,
                 corex_anchor_strength=1.6,
                 corex_anchor_words=None,
                 top_words: int = 10,
                 n_sentences: int = 10,
                 text_column: str = 'text',
                 text_summary: TextSummary = None,
                 chester_collector=None):
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
    :param text_column: the text column
    :param corex_anchor_words: corex anchore words (for semi supervised version)
    :param corex_anchor_strength: anchore strength
    :param text_summary: text sentiment, summary and other requirements
    :param chester_collector: collector to track stats
    """

    rc = ReportCollector(REPORT_PATH)
    is_clean_col_exists = True if f"clean_{text_column}" in df.columns else False
    if is_clean_col_exists:
        modified_df = df.drop(columns=[text_column], axis=1). \
            rename(columns={f'clean_{text_column}': text_column})

    print_analyze_message()
    if data_quality:
        print(data_quality_message)
        rc.save_text("Data quality stats:")
        if is_clean_col_exists:
            print(analyze_text_stats(modified_df, text_column), "\n")
        else:
            print(analyze_text_stats(df, text_column), "\n")
    if common_words:
        rc.save_text(common_words_message)
        print(common_words_message)

        if is_clean_col_exists:
            print("\t", cw.most_common_words(modified_df, common_words=top_words, text_column=text_column))
        else:
            print("\t", cw.most_common_words(df, common_words=top_words, text_column=text_column))

        print("\n")
    if create_wordcloud:
        print(word_cloud_message)

        if is_clean_col_exists:
            create_word_cloud(modified_df, text_column=text_column)
        else:
            create_word_cloud(df, text_column=text_column)

    sentiment = text_summary.is_sentiment or True
    if sentiment:
        print(sentiment_analysis_message)
        df = df.copy()
        modified_df = modified_df.copy()
        if is_clean_col_exists:
            df = analyze_sentiment(modified_df, text_column=text_column)
        else:
            df = analyze_sentiment(df, text_column=text_column)
        chester_collector["sentiment df"] = df
        print(report_sentiment_stats(df), "\n")
        print()
        plot_sentiment_scores(df)

    if corex_topics:
        print(corex_topic_message)
        if is_clean_col_exists:
            corex.plot_corex_wordcloud(modified_df, n_topics=corex_topics_num, top_words=top_words,
                                       text_column=text_column, corex_anchor_words=corex_anchor_words,
                                       corex_anchor_strength=corex_anchor_strength)
        else:
            corex.plot_corex_wordcloud(df, n_topics=corex_topics_num, top_words=top_words, text_column=text_column,
                                       corex_anchor_words=corex_anchor_words,
                                       corex_anchor_strength=corex_anchor_strength
                                       )

    if key_sentences:
        print(key_sentences_message)

        if is_clean_col_exists:
            print(extract_key_sentences(modified_df, n_sentences=n_sentences, text_column=text_column))
        else:
            print(extract_key_sentences(df, n_sentences=n_sentences, text_column=text_column))

    if kewords_extraction:
        print(kws_message)

        if is_clean_col_exists:
            full_text = '. '.join(modified_df[text_column])
        else:
            full_text = '. '.join(df[text_column])

        kws_rake = kw_extract.RakeKeywordExtractor()
        print("\t", kws_rake.extract(text=full_text))

    return chester_collector


def analyze_text_df(text_analyzer: TextAnalyzer):
    df = text_analyzer.df
    text_analyzer.chester_collector = analyze_text(df=df, text_column=text_analyzer.text_column,
                                                   create_wordcloud=text_analyzer.create_wordcloud,
                                                   corex_topics=text_analyzer.corex_topics,
                                                   key_sentences=text_analyzer.key_sentences,
                                                   common_words=text_analyzer.common_words,
                                                   data_quality=text_analyzer.data_quality,
                                                   corex_topics_num=text_analyzer.corex_topics_num,
                                                   top_words=text_analyzer.top_words,
                                                   n_sentences=text_analyzer.n_sentences,
                                                   corex_anchor_strength=text_analyzer.corex_anchor_strength,
                                                   corex_anchor_words=text_analyzer.corex_anchor_words,
                                                   text_summary=text_analyzer.text_summary,
                                                   chester_collector=text_analyzer.chester_collector,
                                                   )
