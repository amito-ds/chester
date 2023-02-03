from wordcloud import WordCloud
import matplotlib.pyplot as plt


# define a function that takes a dataframe and creates a word cloud from the 'text' column
def create_word_cloud(data):
    # join the texts into a single string
    text = ' '.join(data['text'])

    # create a WordCloud object
    wordcloud = WordCloud().generate(text)

    # plot the word cloud
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
