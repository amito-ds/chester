import pandas as pd
from nltk.corpus import brown

from cleaning import *
from data_quality import analyze_text_data
from preprocessing import preprocess_text
from text_analyzer import common_words
from text_analyzer.word_cloud import create_word_cloud
from util import get_stopwords

if __name__ == '__main__':
    import subprocess

    # Read the contents of the requirements.txt file
    with open('requirements.txt', 'r') as f:
        requirements = f.read()

    # Split the requirements string into a list of package names
    package_names = requirements.split('\n')

    # Upgrade each package to the latest version
    for package_name in package_names:
        subprocess.run(['pip', 'install', '--upgrade', package_name])
    #
    # brown_sent = brown.sents(categories='news')[:100]
    # brown_sent = [' '.join(x) for x in brown_sent]
    # df = pd.DataFrame({'text': brown_sent})
    #
    # # Clean the text column
    # df['text'] = df['text'].apply(lambda x: clean_text(x,
    #                                                    remove_stopwords_flag=True,
    #                                                    stopwords=get_stopwords()))
    #
    # # preprocess the text column
    # df['text'] = df['text'].apply(lambda x: preprocess_text(x, stem_flag=False))
    #
    # print(df['text'])
    #
    # # basic stats
    # analyze_text_data(df)
    #
    # # use the get_most_common_words function to get the list of most common words
    # most_common_words = common_words.most_common_words(df, n=10)
    #
    # # word cloud
    # create_word_cloud(df)

    # print(extractive_summarization(df))
