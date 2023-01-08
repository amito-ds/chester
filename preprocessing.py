import nltk
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer
from nltk.stem import WordNetLemmatizer


def get_stemmer(name):
    if name == "porter":
        return PorterStemmer()
    elif name == "snowball":
        return SnowballStemmer("english")
    elif name == "lancaster":
        return LancasterStemmer()
    else:
        raise ValueError(f"Invalid stemmer name: {name}")


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


def pos_tag(text):
    tokens = tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    return pos_tags


def stem(words, stemmer=PorterStemmer()):
    stemmed_words = [stemmer.stem(word) for word in words]
    return stemmed_words
