from nltk import WordNetLemmatizer

from cleaning import *
import nltk

# Load the list of stopwords

import nltk

from preprocessing import lemmatize, get_stemmer, preprocess_text


def extract_nouns(text):
  pos_tags = nltk.pos_tag(nltk.word_tokenize(text))
  nouns = [word for word, pos in pos_tags if pos == "NN"]
  return nouns


if __name__ == '__main__':
    text = "ef remove_whitespace(text: str) -> str:"
    cleaned_text = preprocess_text(text, lemmatize_flag=False, stem_flag=True, stemmer=get_stemmer("snowball"))
    print(cleaned_text)

