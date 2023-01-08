import pandas as pd
from nltk.corpus import brown

from cleaning import clean_text
from preprocessing import preprocess_text
from text_analyzer.corex_topics import get_top_words, plot_corex_wordcloud
from util import get_stopwords
import gensim
from gensim import corpora

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
    # topics = get_top_words(df, top_words=5, n_topics=10)
    top_words_list = get_top_words(df, 10, 5)
    print(top_words_list)

    # plot_corex_wordcloud(df)

    # summarize_topics(topics)
    #
    # # basic stats
    # analyze_text_data(df)
    #
    # # use the get_most_common_words function to get the list of most common words
    # most_common_words = common_words.most_common_words(df, n=10)
    # print(most_common_words)
    #
    # # word cloud
    # create_word_cloud(df)

    # print(extractive_summarization(df))

