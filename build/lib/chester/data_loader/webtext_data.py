import nltk
import pandas as pd
from nltk.corpus import webtext

nltk.download('webtext', quiet=True)
nltk.download('punkt', quiet=True)


def webtext_to_df(full_text: str) -> pd.DataFrame:
    # Separate the chat logs into individual messages
    text_rows = full_text.split("\n")
    # Create a dataframe with text rows
    df = pd.DataFrame({'text': text_rows})
    return df


def load_data_pirates():
    poc = webtext.raw('pirates.txt')
    return webtext_to_df(poc)


def load_data_king_arthur():
    king_arthur = webtext.raw('grail.txt')
    return webtext_to_df(king_arthur)


def load_data_chat_logs():
    chat_logs = webtext.raw('singles.txt')
    return webtext_to_df(chat_logs)
