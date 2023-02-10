from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns


def most_common_words(data, common_words=10, text_column='text'):
    # create a Counter object to store the word counts
    word_counts = Counter()

    # iterate over each row in the data
    for index, row in data.iterrows():
        # split the text into a list of words
        words = row[text_column].split()

        # add the words to the Counter
        word_counts.update(words)

    # sort the Counter by value (count) in descending order
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    # create a barplot using seaborn
    plt.figure(figsize=(11, 11))
    plt.rcParams.update({'font.size': 25})
    sns.barplot(x=[t[0] for t in sorted_word_counts[:common_words]],
                y=[t[1] for t in sorted_word_counts[:common_words]])
    plt.xlabel('Word')
    plt.ylabel('Count')
    plt.title('Most Common Words')
    plt.show()

    # return the sorted list of tuples (word, count)
    return sorted_word_counts[:common_words]
