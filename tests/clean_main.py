import nltk

from cleaning import *
from typing import List

# Load the list of stopwords


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Load the reviews dataset
    reviews = [
        "I absolutely love this product! It has changed my life!",
        "I'm not a huge fan of this product. It's just okay.",
        "This product is terrible! I would never buy it again.",
        "I've been using this product for a few months now, and I have to say that I'm really impressed. It's made a big difference in my daily routine.",
        "I was skeptical about this product at first, but after using it for a while, I've come to really enjoy it. It's definitely worth the investment.",
    ]

    # Load the list of stopwords
    stopwords = nltk.corpus.stopwords.words("english")

    # Clean the reviews
    cleaned_reviews = []
    for review in reviews:
        cleaned_review = clean_text(review,
                                    remove_punctuation_flag=True,
                                    remove_numbers_flag=True,
                                    remove_whitespace_flag=True,
                                    lowercase_flag=True,
                                    remove_stopwords_flag=True,
                                    stopwords=stopwords,
                                    remove_accented_characters_flag=True,
                                    remove_special_characters_flag=True)
        cleaned_reviews.append(cleaned_review)

    print(cleaned_reviews)
