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


# get port stemmer
port_stemmer = PorterStemmer()
# get word net lemmatizer
world_ner_lemmatizer = WordNetLemmatizer()


def stem(words, stemmer=port_stemmer):
    stemmed_words = [stemmer.stem(word) for word in words]
    return stemmed_words


def preprocess_text(text, stemmer=None, lemmatizer=None, stem_flag=False, lemmatize_flag=False, tokenize_flag=True,
                    pos_tag_flag=False):
    if stem_flag and lemmatize_flag:
        raise ValueError("Both stemmer and lemmatizer cannot be applied. Please choose one.")

    if tokenize_flag:
        tokens = tokenize(text)
    else:
        tokens = text

    if pos_tag_flag:
        pos_tags = pos_tag(tokens)
    else:
        pos_tags = tokens
    if stem_flag:
        stemmer = stemmer if stemmer is not None else PorterStemmer
        stemmed_words = stem(pos_tags, stemmer=stemmer)
    else:
        stemmed_words = pos_tags

    if lemmatize_flag:
        lemmatized_words = lemmatize(stemmed_words,
                                     lemmatizer=lemmatizer if lemmatizer is not None else world_ner_lemmatizer)
    else:
        lemmatized_words = stemmed_words

    return ' '.join(lemmatized_words)
