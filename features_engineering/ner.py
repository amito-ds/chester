import pandas as pd
import spacy
from nltk.corpus import brown

from cleaning.cleaning import clean_text
from preprocessing.preprocessing import preprocess_text
from util import get_stopwords
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from pretrained_models.local_models import ner_model_path


# def load_ner_model(model_name):
#     # Load the specified NER model
#     nlp = spacy.load(model_name)
#
#     return nlp

def load_ner_model(path: str = ner_model_path):
    return spacy.load(path)


def classify_text(text, nlp=load_ner_model()):
    # Use the NER model to identify named entities in the text
    doc = nlp(text)
    named_entities = [ent.text for ent in doc.ents]

    return named_entities


def get_ner_bow_embedding(training_data: pd.DataFrame, class_col="class", embedding_size=100,
                          test: pd.DataFrame = None):
    bow_col_prefix = "ner_bow"
    # Join the named entities in each element of the class_col column into a single string
    named_entities_str = training_data[class_col].apply(lambda x: " ".join(x))

    # Vectorize the named entities using a BOW model with a specified number of features
    vectorizer = CountVectorizer(max_features=embedding_size)
    X = normalize(vectorizer.fit_transform(named_entities_str))

    feature_names = list(vectorizer.vocabulary_.keys())
    # Create a DataFrame for the training data
    train_embedding_df = pd.DataFrame(X.todense(), columns=[f"{bow_col_prefix}_{word}" for word in feature_names])

    if test is not None:
        # Join the named entities in each element of the class_col column in the test data into a single string
        test_named_entities_str = test[class_col].apply(lambda x: " ".join(x))

        # Vectorize the named entities in the test data using the same BOW model
        test_X = normalize(vectorizer.transform(test_named_entities_str))

        # Create a DataFrame for the test data
        test_embedding_df = pd.DataFrame(test_X.todense(),
                                         columns=[f"{bow_col_prefix}_{word}" for word in feature_names])

        return train_embedding_df, test_embedding_df
    else:
        return train_embedding_df


# Define a function to create an embedding for the named entities in text data


if __name__ == '__main__':

    brown_sent = brown.sents(categories='reviews')[:100]
    brown_sent = [' '.join(x) for x in brown_sent]
    df = pd.DataFrame({'text': brown_sent})

    # Clean the text column
    df['text'] = df['text'].apply(lambda x: clean_text(x,
                                                       remove_stopwords_flag=True,
                                                       stopwords=get_stopwords()))

    # preprocess the text column
    df['clean_text'] = df['text'].apply(lambda x: preprocess_text(x, stem_flag=False))

    df["class"] = df["clean_text"].apply(classify_text)

    # Print the resulting classification for each text element
    print(df["class"])

    # Calculate the normalized BOW representation for the "texas" class

    bow_nor = get_ner_bow_embedding(df, "class", embedding_size=3)
    print(np.sum(bow_nor ** 2, axis=1))
    print(bow_nor[0:10])

    # Print the resulting DataFrame
    # print(df)d

    # Create an embedding for the "class" column of the DataFrame

    # Calculate the category counts
    # class_counts = df["class"].value_counts()
    #
    # # Plot the category counts as a bar chart
    # plt.bar(class_counts.index, class_counts.values)
    # plt.xlabel("Category")
    # plt.ylabel("Count")
    # plt.title("Category Statistics")
    # plt.show()
