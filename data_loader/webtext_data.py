import pandas as pd
from nltk.corpus import webtext


def get_chat_logs():
    chat_logs = webtext.raw('singles.txt')

    # List the file ids in the webtext corpus

    # Separate the chat logs into individual messages
    logs = chat_logs.split("\n")
    # Create an empty dataframe with a column named 'text'
    df = pd.DataFrame(columns=['text'])

    # Iterate over the logs and add each log as a new row
    for log in logs:
        df = df.append({'text': log}, ignore_index=True)

    return df
