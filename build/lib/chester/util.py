import re
import unicodedata

import nltk
from nltk.corpus import stopwords


def get_stopwords():
    nltk.download('stopwords', quiet=True)
    return stopwords.words('english')


def remove_html(text: str) -> str:
    """Remove HTML tags from a string.

    Args:
        text: The input string.

    Returns:
        The input string with HTML tags removed.
    """
    return re.sub(r'<[^>]*>', '', text)


def remove_accents(text: str) -> str:
    """Remove accents from characters in a string.

    Args:
        text: The input string.

    Returns:
        The input string with accents removed.
    """
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    return text


def expand_contractions(text: str, contraction_mapping: dict) -> str:
    """Expand contractions in a string.

    Args:
        text: The input string.
        contraction_mapping: A dictionary mapping contractions to their expanded form.

    Returns:
        The input string with contractions expanded.
    """
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) if contraction_mapping.get(
            match) else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


contractions_dict = {
    "ain't": "am not / are not / is not / has not / have not",
    "aren't": "are not / am not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had / he would",
    "he'd've": "he would have",
    "he'll": "he shall / he will",
    "he'll've": "he shall have / he will have",
    "he's": "he has / he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how has / how is / how does",
    "I'd": "I had / I would",
    "I'd've": "I would have",
    "I'll": "I shall / I will",
    "I'll've": "I shall have / I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it had / it would",
    "it'd've": "it would have",
    "it'll": "it shall / it will",
    "it'll've": "it shall have / it will have",
    "it's": "it has / it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock"}


def remove_non_ascii(text):
    """Remove non-ASCII characters from the given text."""
    text = ''.join([c for c in text if 0 < ord(c) < 127])
    return text


def remove_numbers(text):
    """Remove numbers from the given text."""
    text = ''.join([c for c in text if not c.isdigit()])
    return text


REPORT_PATH = "report.txt"


class ReportCollector:
    def __init__(self, report_path=REPORT_PATH):
        self.REPORT_PATH = report_path

    def save_text(self, text):
        text = str(text)
        with open(self.REPORT_PATH, 'r') as f:
            contents = f.read()
        with open(self.REPORT_PATH, 'w') as f:
            f.write(contents)
            f.write(text + '\n')

    def save_pandas(self, df, text):
        text = str(text)
        with open(self.REPORT_PATH, 'r') as f:
            contents = f.read()
        with open(self.REPORT_PATH, 'w') as f:
            f.write(contents)
            f.write(text + '\n')
            f.write(df.to_string() + '\n\n')

    def save_object(self, obj, text):
        text = str(text)
        with open(self.REPORT_PATH, 'r') as f:
            contents = f.read()
        with open(self.REPORT_PATH, 'w') as f:
            f.write(contents)
            f.write(text + '\n')
            f.write(str(obj) + '\n\n')


class RedirectedStdout:
    def __init__(self, file):
        self.file = file

    def write(self, message):
        self.file.write(message)

    # def drop_duplicates_columns()


def remove_prefix_suffix(string, prefix, suffix):
    """
    Remove prefix and suffix from string and return the resulting string.

    Parameters:
    string (str): The input string from which the prefix and suffix will be removed.
    prefix (str): The prefix to be removed from the input string.
    suffix (str): The suffix to be removed from the input string.

    Returns:
    str: The input string with the prefix and suffix removed.
    """
    if string.startswith(prefix):
        string = string[len(prefix):]
    if string.endswith(suffix):
        string = string[:-len(suffix)]
    return string
