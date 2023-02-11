import matplotlib.pyplot as plt
from wordcloud import WordCloud


# define a function that takes a dataframe and creates a word cloud from the 'text' column
def create_word_cloud(data, text_column='text'):
    # join the texts into a single string
    text = ' '.join(data[text_column])

    # create a WordCloud object
    wordcloud = WordCloud().generate(text)

    # plot the word cloud
    plt.figure(figsize=(14, 14))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    plt.close()
