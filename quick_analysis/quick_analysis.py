import os
import sys

path = os.path.abspath("TCAP")
sys.path.append(path)
from cleaning import cleaning
from data_loader.webtext_data import *
from features_engineering.fe_main import get_embeddings
from preprocessing.preprocessing import preprocess_df_text
from util import get_stopwords
from nltk import PorterStemmer, WordNetLemmatizer

from text_analyzer import smart_text_analyzer
from termcolor import colored

from termcolor import colored


def print_step_message(step_name: str):
    message = "#### Starting {} step ####".format(step_name)
    print("\n" + "#" * len(message) + "\n", end="")
    print(message)
    print("\n" + "#" * len(message) + "\n", end="\n")


# get stop words
stop_words = get_stopwords()

# get port stemmer
port_stemmer = PorterStemmer()
# get word net lemmatizer
world_ner_lemmatizer = WordNetLemmatizer()

# Default cleaning options
default_cleaning_options = {
    'remove_punctuation_flag': True,
    'remove_numbers_flag': True,
    'remove_whitespace_flag': True,
    'lowercase_flag': True,
    'remove_stopwords_flag': False,
    'stopwords': stop_words,
    'remove_accented_characters_flag': True,
    'remove_special_characters_flag': True,
    'remove_html_tags_flag': True
}

# Default preprocessing options
default_preprocessing_options = {
    'stemmer': port_stemmer,
    'lemmatizer': world_ner_lemmatizer,
    'stem_flag': True,
    'lemmatize_flag': False,
    'tokenize_flag': True,
    'pos_tag_flag': False
}

# Default analysis options
default_analysis_options = {
    'create_wordcloud': True,
    'corex_topics': True,
    'key_sentences': True,
    'common_words': True,
    'sentiment': True,
    'data_quality': True,
    'corex_topics_num': 5,
    'top_words': 10,
    'n_sentences': 5
}

# Default embeddings options
default_embeddings_options = {
    'corex': True,
    'corex_dim': 100,
    'tfidf': True,
    'tfidf_dim': 100,
    'bow': True,
    'bow_dim': 100,
    'ngram_range': (1, 1)
}


def process_text(train_data: pd.DataFrame,
                 test_data: pd.DataFrame = None,
                 test_prop=0.2,
                 text_column: str = 'text',
                 label_column: str = None,
                 cleaning_options=None,
                 preprocessing_options: dict = None,
                 analysis_options: dict = None,
                 embeddings_options: dict = None):
    cleaning_options = cleaning_options or default_cleaning_options
    preprocessing_options = preprocessing_options or default_preprocessing_options
    analysis_options = analysis_options or default_analysis_options
    embeddings_options = embeddings_options or default_embeddings_options

    # Clean the data
    print_step_message("cleaning")
    train_data['clean_text'] = cleaning.clean_df_text(train_data[text_column], cleaning_options)

    # preprocess the text column
    print_step_message("preprocessing")
    train_data['clean_text'] = preprocess_df_text(train_data['clean_text'], preprocessing_options)

    if test_data is None and test_prop > 0:
        test_data = train_data.sample(frac=test_prop, random_state=42)
    elif test_data is not None and test_prop <= 0:
        test_data = None
    else:
        # Clean the data
        print_step_message("cleaning test set")
        test_data['clean_text'] = cleaning.clean_df_text(test_data[text_column], cleaning_options)
        # preprocess the test data
        print_step_message("preprocessing test set")
        test_data['clean_text'] = preprocess_df_text(test_data['clean_text'], preprocessing_options)

    # basic stats
    print_step_message("analyzer")
    smart_text_analyzer.analyze_text(train_data, **analysis_options)

    # create bow, itf idf and corex embedding
    print_step_message("embedding")
    train_embedding = get_embeddings(training_data=train_data, **embeddings_options)
    if test_data is not None:
        print_step_message("embedding for test set")
        test_embedding = get_embeddings(training_data=test_data, **embeddings_options)
    else:
        test_embedding = None

    return train_embedding, test_embedding


if __name__ == '__main__':
    df = load_data_chat_logs()
    df_embedding, _ = process_text(train_data=df, test_data=None)
    # print(df_embedding.shape)
