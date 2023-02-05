import nltk
import pandas as pd
import seaborn as sns

# nltk.download('punkt')
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt


def calculate_text_metrics(text):
    metrics = {'text_length': len(text), 'num_sentences': len(nltk.sent_tokenize(text))}
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


def calculate_text_column_metrics(df):
    df = df.copy()
    df['text_metrics'] = df['text'].apply(calculate_text_metrics)
    words = df['text'].str.split(expand=True).stack().unique()
    num_unique_words = len(words)
    df = json_normalize(df['text_metrics'])
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


def analyze_text_data(df):
    df, num_unique_words = calculate_text_column_metrics(df)
    report = create_report(df, num_unique_words)
    print(report)
    plot_text_length_and_num_words(df)
