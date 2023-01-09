from typing import Dict, Union

import pandas as pd
from nltk.corpus import brown
from textblob import TextBlob

from cleaning import clean_text
from preprocessing import preprocess_text
from util import get_stopwords

import pandas as pd
from textblob import TextBlob
import altair as alt


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

    # return results as a dictionary
    return {
        'positive_count': pos_count,
        'negative_count': neg_count,
        'neutral_count': neutral_count,
        'positive_percent': pos_percent,
        'negative_percent': neg_percent,
        'neutral_percent': neutral_percent
    }


def analyze_sentiment(df):
    # Add a new column to the DataFrame with the sentiment of each text
    df['sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)

    return df


import matplotlib.pyplot as plt


def plot_sentiment_scores(sentiment_df):
    # Get the count of each sentiment score
    counts = sentiment_df['sentiment'].value_counts()

    # Set the x and y values for the bar chart
    x = counts.index
    y = counts.values

    # Create the bar chart
    plt.bar(x, y)
    plt.xlabel('Sentiment Score')
    plt.ylabel('Count')
    plt.title('Sentiment Scores')

    # Show the plot
    plt.show()


if __name__ == '__main__':
    brown_sent = brown.sents(categories='reviews')[:100]
    brown_sent = [' '.join(x) for x in brown_sent]
    df = pd.DataFrame({'text': brown_sent})

    # Clean the text column
    df['text'] = df['text'].apply(lambda x: clean_text(x,
                                                       remove_stopwords_flag=True,
                                                       stopwords=get_stopwords()))

    # preprocess the text column
    df['clean_text'] = df['text'].apply(lambda x: preprocess_text(x, stem_flag=False))

    sentiment_df = analyze_sentiment(df)
    print(report_sentiment_stats(sentiment_df))
    plot_sentiment_scores(sentiment_df)
