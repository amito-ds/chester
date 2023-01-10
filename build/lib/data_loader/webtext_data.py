import pandas as pd
from nltk.corpus import webtext


def webtext_to_df(full_text: str) -> pd.DataFrame:
    # List the file ids in the webtext corpus

    # Separate the chat logs into individual messages
    text_rows = full_text.split("\n")
    # Create an empty dataframe with a column named 'text'
    df = pd.DataFrame(columns=['text'])

    # Iterate over the logs and add each log as a new row
    for text_row in text_rows:
        df = df.append({'text': text_row}, ignore_index=True)

    return df


def load_pirates():
    poc = webtext.raw('pirates.txt')
    return webtext_to_df(poc)


def load_king_arthur():
    king_arthur = webtext.raw('grail.txt')
    return webtext_to_df(king_arthur)


def get_chat_logs():
    chat_logs = webtext.raw('singles.txt')
    return webtext_to_df(chat_logs)
