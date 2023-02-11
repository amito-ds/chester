from typing import Dict, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from prettytable import PrettyTable
from textblob import TextBlob


def report_sentiment_stats(sentiment_df: pd.DataFrame) -> Dict[str, Union[int, float]]:
    # count number of positive, negative and neutral sentiments
    pos_count = len(sentiment_df[sentiment_df['sentiment'] > 0])
    neg_count = len(sentiment_df[sentiment_df['sentiment'] < 0])
    neutral_count = len(sentiment_df[sentiment_df['sentiment'] == 0])
    # calculate percentage of positive, negative and neutral sentiments
    total_count = pos_count + neg_count + neutral_count
    pos_percent = pos_count / total_count * 100
    neg_percent = neg_count / total_count * 100
    neutral_percent = neutral_count / total_count * 100

    # create a table
    table = PrettyTable()
    table.field_names = ["Sentiment", "Count", "Percentage"]
    table.add_row(["Positive", pos_count, f"{pos_percent:.1f}%"])
    table.add_row(["Negative", neg_count, f"{neg_percent:.1f}%"])
    table.add_row(["Neutral", neutral_count, f"{neutral_percent:.1f}%"])

    return table.get_string(title="Sentiment Statistics")


def analyze_sentiment(df, text_column='text'):
    # Add a new column to the DataFrame with the sentiment of each text
    df['sentiment'] = df[text_column].apply(lambda x: TextBlob(x).sentiment.polarity)

    return df


def plot_sentiment_scores(sentiment_df):
    # Create a histogram of the sentiment scores using Seaborn
    ax = sns.histplot(data=sentiment_df, x='sentiment', kde=False)
    ax.set(xlabel='Sentiment Score', ylabel='Count', title='Sentiment Scores')
    plt.figure(figsize=(14, 14))
    plt.rcParams.update({'font.size': 14})
    plt.show()
