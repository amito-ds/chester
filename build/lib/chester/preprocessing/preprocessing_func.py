import nltk
import pandas as pd
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer
from nltk.stem import WordNetLemmatizer

from chester.util import ReportCollector, REPORT_PATH


def get_stemmer(name=None):
    if name == "porter":
        return PorterStemmer()
    elif name == "snowball":
        return SnowballStemmer("english")
    elif name == "lancaster":
        return LancasterStemmer()
    else:
        return PorterStemmer()


def lemmatize(text, lemmatizer=WordNetLemmatizer()):
    lemmas = []
    pos_tags = nltk.pos_tag(text)
    for word, pos_tag in pos_tags:
        pos = get_wordnet_pos(pos_tag)
        lemma = lemmatizer.lemmatize(word, pos)
        lemmas.append(lemma)
    return lemmas


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    return tokens


# get port stemmer
port_stemmer = PorterStemmer()
# get word net lemmatizer
world_ner_lemmatizer = WordNetLemmatizer()


def stem(words, stemmer=port_stemmer):
    stemmed_words = [stemmer.stem(word) for word in words]
    return stemmed_words


def preprocess_df_text(text_column, preprocessing_options: dict):
    print_preprocessing_options(**preprocessing_options)
    return text_column.apply(lambda x: preprocess_text(x, **preprocessing_options))


def print_preprocessing_options(stemmer=None, lemmatizer=None, stem_flag=False, lemmatize_flag=False,
                                tokenize_flag=True):
    print("Preprocessing step:")
    if stem_flag:
        print("Stemming text using {} stemmer".format(stemmer.__class__.__name__))
    if lemmatize_flag:
        print("Lemmatizing text using {} lemmatizer".format(lemmatizer.__class__.__name__))
    if tokenize_flag:
        print("Tokenizing text")


def preprocess_text(text,
                    stemmer=None,
                    lemmatizer=None,
                    stem_flag=False,
                    lemmatize_flag=False,
                    tokenize_flag=True):
    if stem_flag and lemmatize_flag:
        raise ValueError("Both stemmer and lemmatizer cannot be applied. Please choose one.")

    if tokenize_flag:
        tokens = tokenize(text)
    else:
        tokens = text

    if stem_flag:
        stemmer = stemmer if stemmer is not None else PorterStemmer()
        stemmed_words = stem(tokens, stemmer=stemmer)
    else:
        stemmed_words = tokens

    if lemmatize_flag:
        lemmatized_words = lemmatize(stemmed_words,
                                     lemmatizer=lemmatizer if lemmatizer is not None else world_ner_lemmatizer)
    else:
        lemmatized_words = stemmed_words

    return ' '.join(lemmatized_words)


class TextPreprocessor:
    def __init__(self,
                 df: pd.DataFrame = None,
                 text_column: str = 'text',
                 stemmer=None,
                 lemmatizer=None,
                 stem_flag=True,
                 lemmatize_flag=False,
                 tokenize_flag=True):
        self.df = df
        self.text_column = text_column
        self.stemmer = stemmer
        self.lemmatizer = lemmatizer
        self.stem_flag = stem_flag
        self.lemmatize_flag = lemmatize_flag
        self.tokenize_flag = tokenize_flag

    def generate_report(self):
        report_str = ""
        if self.stem_flag:
            if self.stemmer:
                if self.stemmer == "snowball":
                    report_str += "Using Snowball stemmer, "
                else:
                    report_str += f"Using {self.stemmer} stemmer, "
            else:
                report_str += "Stemming text, "
        if self.lemmatize_flag:
            if self.lemmatizer:
                report_str += f"Using {self.lemmatizer} lemmatizer, "
            else:
                report_str += "Lemmatizing text, "
        if self.tokenize_flag:
            report_str += "Tokenizing text, "
        if report_str:
            report_str = report_str[:-2]
            title_to_print = f"The following preprocessing steps will be applied to the column '{self.text_column}': {report_str}."
            print(title_to_print)
            rc = ReportCollector(REPORT_PATH)
            rc.save_text(title_to_print)
        else:
            print("No preprocessing steps selected.")


def preprocess_text_df(text_preprocessor: TextPreprocessor):
    df = text_preprocessor.df
    text_column = text_preprocessor.text_column

    df[text_column] = df[text_column].apply(lambda x: preprocess_text(
        x,
        stemmer=text_preprocessor.stemmer,
        lemmatizer=text_preprocessor.lemmatizer,
        stem_flag=text_preprocessor.stem_flag,
        lemmatize_flag=text_preprocessor.lemmatize_flag,
        tokenize_flag=text_preprocessor.tokenize_flag,
    ))
    return df
