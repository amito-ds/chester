import re
import string
import unicodedata
from typing import List

import pandas as pd

from chester.util import get_stopwords, ReportCollector, REPORT_PATH


def remove_punctuation(text: str) -> str:
    """Remove all punctuation from the given text."""
    return str(text).translate(str(text).maketrans("", "", string.punctuation))


def remove_numbers(text: str) -> str:
    """Remove all numbers from the given text."""
    return ''.join(c for c in text if not c.isdigit())


def remove_whitespace(text: str) -> str:
    """Remove excess whitespace from the given text."""
    return ' '.join(str(text).split())


def remove_empty_lines(text: str):
    """Remove excess empty lines from the given text."""
    lines = str(text).splitlines()
    cleaned_text = '\n'.join([line for line in lines if line.strip()])
    return cleaned_text


def lowercase(text: str) -> str:
    """Convert the given text to lowercase."""
    return text.lower()


def remove_stopwords(text: str, stopwords: List[str]) -> str:
    """Remove common words that do not contribute to the meaning of the text.

    stopwords: a list of words to remove from the text.
    """
    words = text.split()
    cleaned_words = [word for word in words if word not in stopwords]
    return ' '.join(cleaned_words)


def remove_accented_characters(text: str) -> str:
    """Remove accented characters from the given text."""
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))


def remove_special_characters(text: str) -> str:
    """Remove special characters from the given text."""
    return re.sub(r'[^\w\s]', '', text)


def remove_html_tags(text: str) -> str:
    """Remove HTML tags from the given text."""
    return re.sub(r'<[^<]+?>', '', text)


def clean_df_text(text_column, cleaning_options: dict):
    print_cleaning_options(**cleaning_options)
    return text_column.apply(lambda x: clean_text(x, **cleaning_options))


def print_cleaning_options(
        stopwords: str,
        remove_punctuation_flag: bool = True,
        remove_numbers_flag: bool = True,
        remove_whitespace_flag: bool = True,
        lowercase_flag: bool = True,
        remove_stopwords_flag: bool = False,
        remove_accented_characters_flag: bool = True,
        remove_special_characters_flag: bool = True,
        remove_html_tags_flag: bool = True):
    if remove_punctuation_flag:
        print("Removing punctuation")
    if remove_numbers_flag:
        print("Removing numbers")
    if remove_whitespace_flag:
        print("Removing whitespaces")
    if lowercase_flag:
        print("Lowercasing text")
    if remove_stopwords_flag:
        print("Removing stop words")
    if remove_accented_characters_flag:
        print("Removing accented characters")
    if remove_special_characters_flag:
        print("Removing special characters")
    if remove_html_tags_flag:
        print("Removing html tags")


def clean_text(text: str,
               remove_punctuation_flag: bool = True,
               remove_numbers_flag: bool = True,
               remove_whitespace_flag: bool = True,
               remove_empty_line_flag: bool = True,
               lowercase_flag: bool = True,
               remove_stopwords_flag: bool = True,
               stopwords: List[str] = None,
               remove_accented_characters_flag: bool = True,
               remove_special_characters_flag: bool = True,
               remove_html_tags_flag: bool = True) -> str:
    """Apply a series of cleaning functions to the given text.

    text: the text to clean.
    remove_punctuation_flag: a flag indicating whether to remove punctuation from the text.
    remove_numbers_flag: a flag indicating whether to remove numbers from the text.
    remove_whitespace_flag: a flag indicating whether to remove excess whitespace from the text.
    remove_empty_line_flag: a flag indicating whether to remove excess empty lines from the text.
    lowercase_flag: a flag indicating whether to convert the text to lowercase.
    remove_stopwords_flag: a flag indicating whether to remove common words that do not contribute to the meaning of the text.
    stopwords: a list of words to remove from the text. Required if remove_stopwords_flag is True.
    remove_accented_characters_flag: a flag indicating whether to remove accented characters from the text.
    remove_special_characters_flag: a flag indicating whether to remove special characters from the text.
    remove_html_tags_flag: a flag indicating whether to remove HTML tags from the text.
    """

    text = str(text)
    if remove_punctuation_flag:
        text = remove_punctuation(text)
    if remove_numbers_flag:
        text = remove_numbers(text)
    if remove_whitespace_flag:
        text = remove_whitespace(text)
    if remove_empty_line_flag:
        remove_empty_lines(text)
    if lowercase_flag:
        text = lowercase(text)
    if remove_stopwords_flag:
        if not stopwords:
            stopwords = get_stopwords()
        text = remove_stopwords(text, stopwords)
    if remove_accented_characters_flag:
        text = remove_accented_characters(text)
    if remove_special_characters_flag:
        text = remove_special_characters(text)
    if remove_html_tags_flag:
        text = remove_html_tags(text)
    return text


from typing import List


class TextCleaner:
    def __init__(self,
                 df: pd.DataFrame = None,
                 text_column: str = 'text',
                 remove_punctuation_flag: bool = True,
                 remove_numbers_flag: bool = True,
                 remove_whitespace_flag: bool = True,
                 remove_empty_line_flag: bool = True,
                 lowercase_flag: bool = True,
                 remove_stopwords_flag: bool = True,
                 stopwords: List[str] = None,
                 remove_accented_characters_flag: bool = True,
                 remove_special_characters_flag: bool = True,
                 remove_html_tags_flag: bool = True):
        self.df = df
        self.text_column = text_column
        self.remove_punctuation_flag = remove_punctuation_flag
        self.remove_numbers_flag = remove_numbers_flag
        self.remove_whitespace_flag = remove_whitespace_flag
        self.remove_empty_line_flag = remove_empty_line_flag
        self.lowercase_flag = lowercase_flag
        self.remove_stopwords_flag = remove_stopwords_flag
        self.stopwords = stopwords
        self.remove_accented_characters_flag = remove_accented_characters_flag
        self.remove_special_characters_flag = remove_special_characters_flag
        self.remove_html_tags_flag = remove_html_tags_flag

    def generate_report(self):
        rc = ReportCollector(REPORT_PATH)
        report_str = ""
        if self.remove_punctuation_flag:
            report_str += "Removing punctuation, "
        if self.remove_numbers_flag:
            report_str += "Removing numbers, "
        if self.remove_whitespace_flag:
            report_str += "Removing whitespaces, "
        if self.remove_empty_line_flag:
            report_str += "Removing empty lines, "
        if self.lowercase_flag:
            report_str += "Lowercasing text, "
        if self.remove_stopwords_flag:
            if self.stopwords:
                report_str += f"Stopwords are given, "
            else:
                report_str += "Removing stopwords, "
        if self.remove_accented_characters_flag:
            report_str += "Removing accented characters, "
        if self.remove_special_characters_flag:
            report_str += "Removing special characters, "
        if self.remove_html_tags_flag:
            report_str += "Removing html tags, "
        if report_str:
            report_str = report_str[:-2]
            title_to_print = f"The following cleaning steps will be applied to clean column '{self.text_column}': {report_str}."
            print(title_to_print)
            rc.save_text(title_to_print)
        else:
            print("No cleaning steps selected.")


def clean_text_df(text_cleaner: TextCleaner) -> pd.DataFrame:
    df = text_cleaner.df
    text_col = text_cleaner.text_column
    df[text_col] = df[text_col].apply(lambda x: clean_text(
        x,
        remove_punctuation_flag=text_cleaner.remove_punctuation_flag,
        remove_numbers_flag=text_cleaner.remove_numbers_flag,
        remove_whitespace_flag=text_cleaner.remove_whitespace_flag,
        remove_empty_line_flag=text_cleaner.remove_empty_line_flag,
        lowercase_flag=text_cleaner.lowercase_flag,
        remove_stopwords_flag=text_cleaner.remove_stopwords_flag,
        stopwords=text_cleaner.stopwords,
        remove_accented_characters_flag=text_cleaner.remove_accented_characters_flag,
        remove_special_characters_flag=text_cleaner.remove_special_characters_flag,
        remove_html_tags_flag=text_cleaner.remove_html_tags_flag
    ))
    return df
