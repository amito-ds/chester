import nltk
import pandas as pd
import seaborn as sns

from pandas.io.json import json_normalize
import matplotlib.pyplot as plt


def calculate_text_metrics(text):
    metrics = {'text_length': len(text), 'num_sentences': len(nltk.sent_tokenize(text))}
    if not text:
        return metrics
    if any(c.isalpha() for c in text):
        metrics['num_characters'] = len([c for c in text if c.isalpha()])
    else:
        metrics['num_characters'] = 0
    words = pd.Series(text.split())
    if len(words) > 0:
        metrics['num_words'] = len(words)
        metrics['num_unique_words'] = len(words.unique())
        metrics['ttr'] = metrics['num_unique_words'] / metrics['num_words']
    else:
        metrics['num_words'] = 0
        metrics['num_unique_words'] = 0
        metrics['ttr'] = 0
    metrics['num_missing'] = int(text == "")
    return metrics


def calculate_text_column_metrics(df, text_column='text'):
    df = df.copy()
    df[f'{text_column}_metrics'] = df[text_column].apply(calculate_text_metrics)
    words = df[text_column].str.split(expand=True).stack().unique()
    num_unique_words = len(words)
    df = pd.json_normalize(df[f'{text_column}_metrics'])
    return df, num_unique_words


def create_report(df, num_unique_words):
    report = ""
    report += f"Number of rows with missing data: {df['num_missing'].sum()}\n"
    report += f"Number of unique words: {num_unique_words}\n"
    report += f"Average type-token ratio: {df['ttr'].mean():.2f}\n"
    report += f"Median type-token ratio: {df['ttr'].median():.2f}\n"
    report += f"Average number of words per text: {df['num_words'].mean():.2f}\n"
    report += f"Median number of words per text: {df['num_words'].median():.2f}\n"
    report += f"Average number of sentences per text: {df['num_sentences'].mean():.2f}\n"
    report += f"Median number of sentences per text: {df['num_sentences'].median():.2f}\n"
    report += f"Average length of text: {df['text_length'].mean():.2f}\n"
    report += f"Median length of text: {df['text_length'].median():.2f}\n"
    return report


def plot_text_length_and_num_words(df):
    sns.set(style="darkgrid")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    sns.histplot(data=df, x='text_length', ax=ax[0])
    sns.histplot(data=df, x='num_words', ax=ax[1])
    fig.suptitle('Text Statistics')
    plt.show()


def analyze_text_stats(df):
    df, num_unique_words = calculate_text_column_metrics(df)
    report = create_report(df, num_unique_words)
    print(report)
    plot_text_length_and_num_words(df)


class TextAnalyzer:
    def __init__(self, df: pd.DataFrame = None, create_wordcloud: bool = True,
                 corex_topics: bool = True, key_sentences: bool = True,
                 common_words: bool = True, sentiment: bool = True,
                 ner_extraction: bool = True, kewords_extraction: bool = True,
                 data_quality: bool = True, corex_topics_num: int = 10,
                 top_words: int = 10, n_sentences: int = 5):
        self.df = df
        self.create_wordcloud = create_wordcloud
        self.corex_topics = corex_topics
        self.key_sentences = key_sentences
        self.common_words = common_words
        self.sentiment = sentiment
        self.data_quality = data_quality
        self.ner_extraction = ner_extraction
        self.kewords_extraction = kewords_extraction
        self.corex_topics_num = corex_topics_num
        self.top_words = top_words
        self.n_sentences = n_sentences
