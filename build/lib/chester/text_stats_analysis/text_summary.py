import heapq
from collections import defaultdict

from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

from chester.cleaning.cleaning_func import clean_text
import pandas as pd


def get_summary_from_df(df, text_column='text', summary_num_sentences=3):
    # Add a new column to the DataFrame with the sentiment of each text
    text = df[text_column].str.cat(sep=". ")
    return summarize_text_textrank(text=text, num_sentences=summary_num_sentences, max_terms=10000)


def get_summary(df, text_columns=None, text_summary=None):
    summary_dict = {}
    # Add a new column to the DataFrame with the sentiment of each text
    if text_columns is None:
        text_columns = ['text']

    if text_summary.id_cols is None:
        for text_col in text_columns:
            text = df[text_col].apply(lambda x: clean_text(
                x,
                remove_punctuation_flag=False,
                remove_numbers_flag=True,
                remove_whitespace_flag=True,
                remove_empty_line_flag=True,
                lowercase_flag=True,
                remove_stopwords_flag=False,
                stopwords=None,
                remove_accented_characters_flag=True,
                remove_special_characters_flag=False,
                remove_html_tags_flag=True
            )).str.cat(sep=". ")  # raw
            summary_dict[text_col] = summarize_text_textrank(text=text,
                                                             num_sentences=text_summary.summary_num_sentences,
                                                             max_terms=text_summary.max_terms)
    else:
        text_columns = list(text_columns)
        # Group the text columns by the ID columns and concatenate the text
        all_cols = list(set(list(text_columns) + list(text_summary.id_cols)))
        aggregated = df[all_cols].groupby(text_summary.id_cols).agg(
            {col: lambda x: '. '.join(clean_text(val,
                                                 remove_punctuation_flag=False,
                                                 remove_numbers_flag=True,
                                                 remove_whitespace_flag=True,
                                                 remove_empty_line_flag=True,
                                                 lowercase_flag=True,
                                                 remove_stopwords_flag=False,
                                                 stopwords=None,
                                                 remove_accented_characters_flag=True,
                                                 remove_special_characters_flag=False,
                                                 remove_html_tags_flag=True) for val in x.dropna()) for col in
             text_columns}).reset_index()
        # Combine id_cols and aggregated text_cols into a single DataFrame
        grouped_text = df[list(text_summary.id_cols)].drop_duplicates().merge(aggregated, on=text_summary.id_cols,
                                                                              how='left')
        grouped_text[text_columns] = grouped_text[text_columns].applymap(
            lambda x: summarize_text_textrank(x, num_sentences=3, max_terms=100))
        summary_dict = grouped_text.set_index(text_summary.id_cols)[text_columns].to_dict('index')

    return summary_dict


def summarize_text_textrank(text, num_sentences, max_terms=100):
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    word_freq = FreqDist(words)
    ranking = defaultdict(int)
    total_words = 0
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                ranking[i] += word_freq[word]
        total_words += len(word_tokenize(sentence))
        if total_words > max_terms:
            num_sentences = i + 1
            break
    top_sentences = heapq.nlargest(num_sentences, ranking, key=ranking.get)
    summary = ' '.join([sentences[i] for i in sorted(top_sentences)])
    return summary
