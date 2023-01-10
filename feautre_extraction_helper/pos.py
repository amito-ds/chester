import pandas as pd
# import spacy
import spacy

#
# def get_pos_embedding(train_df: pd.DataFrame, test_df: pd.DataFrame = None, text_column: str = 'text',
#                       nlp=spacy.load('en_core_web_md')):
#     """
#     Creates a POS embedding for the text data in the given DataFrames.
#     Parameters:
#     - train_df: A DataFrame with the training data.
#     - test_df: A DataFrame with the test data.
#     - text_column: The name of the column in the DataFrames that contains the text data.
#     - nlp: The spacy nlp object to use for POS tagging.
#
#     Returns:
#     - A tuple with the following elements:
#         - A DataFrame with the POS embedding for the training data.
#         - A DataFrame with the POS embedding for the test data (if provided).
#     """
#     # Initialize lists to store the POS tags for each text
#     pos_tags_train = []
#     pos_tags_test = []
#
#     # Iterate through the training texts
#     for text in train_df[text_column]:
#         # Tokenize the text
#         doc = nlp(text)
#
#         # Get the POS tags for each token
#         tags = [token.pos_ for token in doc]
#
#         # Add the tags to the list
#         pos_tags_train.append(tags)
#
#     # Check if test data was provided
#     if test_df is not None:
#         # Iterate through the test texts
#         for text in test_df[text_column]:
#             # Tokenize the text
#             doc = nlp(text)
#
#             # Get the POS tags for each token
#             tags = [token.pos_ for token in doc]
#
#             # Add the tags to the list
#             pos_tags_test.append(tags)
#
#     # Create a DataFrame with the POS tags for the training data
#     train_df = pd.DataFrame(pos_tags_train, columns=[f"pos_{i+1}" for i in range(len(pos_tags_train[0]))])
#
#     # Check if test data was provided
#     if test_df is not None:
#         # Create a DataFrame with the POS tags for the test data
#         test_df = pd.DataFrame(pos_tags_test, columns=[f"pos_{i+1}" for i in range(len(pos_tags_test[0]))])
#
#         return train_df, test_df
#     else:
#         return train_df, None



from nltk.corpus import brown

from cleaning import clean_text
from preprocessing import preprocess_text
from util import get_stopwords


if __name__ == '__main__':
    path = '/Users/amitosi/opt/anaconda3/envs/py39/lib/python3.9/site-packages/en_core_web_md/en_core_web_md-3.3.0'
    nlp = spacy.load(path)
    # nlp = spacy.load('en_core_web_md')
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
    # df['clean_text'] = df['text'].apply(lambda x: preprocess_text(x, stem_flag=False))
    #
    # # Extract the bag of words embedding
    # bow_embedding, _, _ = get_pos_embedding(train_df=df, text_column='clean_text')

    # print(bow_embedding[0:10])


