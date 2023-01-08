import matplotlib.pyplot as plt
from collections import Counter


def most_common_words(data, n=10):
    # create a Counter object to store the word counts
    word_counts = Counter()

    # iterate over each row in the data
    for index, row in data.iterrows():
        # split the text into a list of words
        words = row['text'].split()

        # add the words to the Counter
        word_counts.update(words)

    # sort the Counter by value (count) in descending order
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    # plot the word counts as a bar chart
    plt.bar(range(len(sorted_word_counts[:n])), [t[1] for t in sorted_word_counts[:n]], align='center')
    plt.xticks(range(len(sorted_word_counts[:n])), [t[0] for t in sorted_word_counts[:n]])
    plt.xlabel('Word')
    plt.ylabel('Count')
    plt.title('Most Common Words')
    plt.show()

    # return the sorted list of tuples (word, count)
    return sorted_word_counts[:n]
