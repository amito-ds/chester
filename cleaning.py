import string
import unicodedata
import re
from typing import List


def remove_punctuation(text: str) -> str:
    """Remove all punctuation from the given text."""
    return text.translate(text.maketrans("", "", string.punctuation))


def remove_numbers(text: str) -> str:
    """Remove all numbers from the given text."""
    return ''.join(c for c in text if not c.isdigit())


def remove_whitespace(text: str) -> str:
    """Remove excess whitespace from the given text."""
    return ' '.join(text.split())


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
