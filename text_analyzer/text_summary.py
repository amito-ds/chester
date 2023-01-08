import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


def extractive_summarization(data):
    # define a list of stop words to exclude from the summary
    stop_words = set(stopwords.words('english'))

    # create a TfidfVectorizer object
    vectorizer = TfidfVectorizer(stop_words=stop_words)

    # compute the Tf-idf scores for the texts
    X = vectorizer.fit_transform(data['text'])

    # get the feature names
    feature_names = vectorizer.get_feature_names()

    # get the scores for each text
    scores = X.toarray()

    # create an empty list to store the summary
    summary = []

    # iterate over each text
    for i, text in enumerate(data['text']):
        # split the text into sentences
        sentences = nltk.sent_tokenize(text)

        # create a list to store the scores for each sentence
        sentence_scores = []

        # iterate over each sentence
        for j, sentence in enumerate(sentences):
            # compute the total score for the sentence
            total_score = 0
            for word in nltk.word_tokenize(sentence.lower()):
                if word in feature_names:
                    total_score += scores[i][feature_names.index(word)]
            # add the score for the sentence to the list
            sentence_scores.append(total_score)

        # get the index of the highest-scoring sentence
        highest_scoring_index = sentence_scores.index(max(sentence_scores))

        # add the highest-scoring sentence to the summary
        summary.append(sentences[highest_scoring_index])

    # return the summary
    return summary
