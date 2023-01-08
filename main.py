from cleaning import *
import nltk

# Load the list of stopwords

import nltk

from preprocessing import lemmatize, get_stemmer


def extract_nouns(text):
  pos_tags = nltk.pos_tag(nltk.word_tokenize(text))
  nouns = [word for word, pos in pos_tags if pos == "NN"]
  return nouns


if __name__ == '__main__':
    stemmer = get_stemmer("porter")

    print(stemmer)  # ['run', 'running', 'ran', 'run', 'easily', 'fairly']
