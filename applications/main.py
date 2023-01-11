import pandas as pd
from nltk.corpus import brown

from cleaning.cleaning import *
from preprocessing.preprocessing import preprocess_text
from text_analyzer.corex_topics import get_top_words
from util import get_stopwords

if __name__ == '__main__':
    import sys

    print(sys.path)

    #
    # brown_sent = brown.sents(categories=['reviews'])[:100]
    # brown_sent = [' '.join(x) for x in brown_sent]
    # df = pd.DataFrame({'text': brown_sent})
    #
    # # Clean the text column
    # df['text'] = df['text'].apply(lambda x: clean_text(x,
    #                                                    remove_stopwords_flag=True,
    #                                                    stopwords=get_stopwords()))
    #
    # # preprocess the text column
    # df['clean_text'] = df['text'].apply(lambda x: preprocess_text(x, stem_flag=False))


    # basic stats
    # analyze_text_data(df)

    # use the get_most_common_words function to get the list of most common words
    # most_common_words = common_words.most_common_words(df, n=10)

    # word cloud
    # create_word_cloud(df)

    # corex
    # plot_corex_wordcloud(df)

    # get_top_words(df, 10, 10)

    # print(extractive_summarization(df))
