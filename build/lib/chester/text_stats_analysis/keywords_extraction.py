from __future__ import division

import operator
import string

import nltk

from chester.util import ReportCollector, REPORT_PATH


def isPunct(word):
    return len(word) == 1 and word in string.punctuation


def isNumeric(word):
    try:
        float(word) if '.' in word else int(word)
        return True
    except:
        return False


class RakeKeywordExtractor:

    def __init__(self):
        self.stopwords = set(nltk.corpus.stopwords.words())
        self.top_fraction = 1  # consider top third candidate keywords by score

    def _generate_candidate_keywords(self, sentences):
        phrase_list = []
        for sentence in sentences:
            words = map(lambda x: "|" if x in self.stopwords else x,
                        nltk.word_tokenize(sentence.lower()))
            phrase = []
            for word in words:
                if word == "|" or isPunct(word):
                    if len(phrase) > 0:
                        phrase_list.append(phrase)
                        phrase = []
                else:
                    phrase.append(word)
        return phrase_list

    def _calculate_word_scores(self, phrase_list):
        word_freq = nltk.FreqDist()
        word_degree = nltk.FreqDist()
        for phrase in phrase_list:
            degree = len(list(filter(lambda x: not isNumeric(x), phrase))) - 1
            for word in phrase:
                word_freq.update([word])
                word_degree.update([word])
                # word_degree.inc(word, degree)  # other words
        for word in word_freq.keys():
            word_degree[word] = word_degree[word] + word_freq[word]  # itself
        # word score = deg(w) / freq(w)
        word_scores = {}
        for word in word_freq.keys():
            word_scores[word] = word_degree[word] / word_freq[word]
        return word_scores

    def _calculate_phrase_scores(self, phrase_list, word_scores):
        phrase_scores = {}
        for phrase in phrase_list:
            phrase_score = 0
            for word in phrase:
                phrase_score += word_scores[word]
            phrase_scores[" ".join(phrase)] = phrase_score
        return phrase_scores

    def extract(self, text, max_words=7, top_words=10):
        rc = ReportCollector(REPORT_PATH)

        sentences = nltk.sent_tokenize(text)
        phrase_list = self._generate_candidate_keywords(sentences)
        word_scores = self._calculate_word_scores(phrase_list)
        phrase_scores = self._calculate_phrase_scores(
            phrase_list, word_scores)
        sorted_phrase_scores = sorted(phrase_scores.items(), key=operator.itemgetter(1), reverse=True)
        n_phrases = len(sorted_phrase_scores)
        return_list = truncate_and_limit_keywords(sorted_phrase_scores, max_words, top_words)
        rc.save_object(obj=return_list, text="popular terms extraction:")
        return return_list


def truncate_and_limit_keywords(keywords_list, max_words, top_words):
    truncated_keywords = []
    for keyword in keywords_list[:top_words]:
        text = trim_repeated_words(keyword[0], 1)
        score = keyword[1]
        truncated_text = " ".join(text.split()[:max_words])
        truncated_keywords.append((truncated_text, score))
    return truncated_keywords


def trim_repeated_words(text, max_repeated_words):
    words = text.split()
    new_words = []
    word_count = {}
    for word in words:
        if word in word_count:
            word_count[word] += 1
            if word_count[word] <= max_repeated_words:
                new_words.append(word)
        else:
            word_count[word] = 1
            new_words.append(word)
    return ' '.join(new_words)
