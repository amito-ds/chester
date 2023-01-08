import nltk
import numpy as np
import pandas as pd
from scipy import stats

nltk.download('punkt')


def calculate_text_metrics(text):
    metrics = {'text_length': len(text), 'num_words': len(text.split()), 'num_sentences': len(nltk.sent_tokenize(text)),
               'num_characters': len([c for c in text if c.isalpha()]), 'num_missing': int(text == "")}
    words = pd.Series(text.split()).unique()
    metrics['num_unique_words'] = len(words)
    metrics['ttr'] = metrics['num_unique_words'] / metrics['num_words']
    return metrics


def calculate_text_column_metrics(df):
    df = df.copy()
    df['text_metrics'] = df['text'].apply(calculate_text_metrics)
    words = df['text'].str.split(expand=True).stack().unique()
    num_unique_words = len(words)
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
