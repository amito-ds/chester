from databots.code_health.code_organization import FormatFile
from databots.run import run

# run("Write a python code to clean text, complete:")

# text = text + " to train logistic model on titanic data (load it from openml)"
# text = " to perform EDA on given pandas df. the function eda(df), " \
#        "include state of the art plots. write code example with data of your choice."

# text = " to perform a*b function, include a single syntax bug in the code"
# Write a Python function that


# text = """
# Scrape the website https://en.wikipedia.org/wiki/Python_(programming_language)
# and extract the main content of the page using BeautifulSoup. Then, clean the text and remove all stopwords using NLTK.
# Finally, calculate the term frequency of the remaining words and print the top 10 most frequent words along with their
# frequency.
# """

text = """
Somewhere in this project there an image named dog.jpeg. 
Find the image, then 
convert to grayscale, flip, and then show the image 
"""
run(text)
# file_path = "/Users/amitosi/PycharmProjects/chester/databots/generated_code.py"
# FormatFile(file_path=file_path).run()

# FormatFile(file_path).run()
# execute_code(file_path)
