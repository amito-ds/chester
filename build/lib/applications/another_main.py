import pandas as pd
from nltk.corpus import brown

from cleaning.cleaning import clean_text
from preprocessing.preprocessing import preprocess_text
from text_analyzer import common_words
from text_analyzer.word_cloud import create_word_cloud
from util import get_stopwords

if __name__ == '__main__':
    brown_sent = brown.sents(categories='reviews')[:100]
    brown_sent = [' '.join(x) for x in brown_sent]
    df = pd.DataFrame({'text': brown_sent})

    # Clean the text column
    df['text'] = df['text'].apply(lambda x: clean_text(x,
                                                       remove_stopwords_flag=True,
                                                       stopwords=get_stopwords()))

    # preprocess the text column
    df['text'] = df['text'].apply(lambda x: preprocess_text(x, stem_flag=False))

    # corex topic modeling
    # Example usage
    # top_words_list = get_top_words(df, 5, 4, ngram_range=range(1, 3))

    # plot_corex_wordcloud(df)

    # summarize_topics(topics)
    #
    # # basic stats
    # analyze_text_data(df)
    #
    # # use the get_most_common_words function to get the list of most common words
    most_common_words = common_words.most_common_words(df, n=10)
    print(most_common_words)
    #
    # # word cloud
    create_word_cloud(df)

