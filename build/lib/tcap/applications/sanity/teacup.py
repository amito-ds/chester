from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def find_similar_texts(data, index, K=5):
    # Extract the text from the given index
    original_text = data.loc[index, 'text']

    # Extract the target if it exists
    if 'target' in data.columns:
        original_target = data.loc[index, 'target']
    else:
        original_target = 'N/A'

    # Extract the embeddings by dropping the index, text, and target columns if they exists
    embeddings = data.drop(columns=['index', 'text', 'target'], errors='ignore')

    # Compute cosine similarity between the embeddings
    similarity = cosine_similarity(embeddings)

    # Get the indices of the K most similar texts
    most_similar = similarity[index].argsort()[-K:][::-1]

    # Print the original text and target
    print(f'Original text: {original_text}')
    print(f'Original target: {original_target}')
    print()

    # Print the most similar texts
    for i in most_similar:
        text = data.loc[i, 'text']
        target = 'N/A' if 'target' not in data.columns else data.loc[i, 'target']
        print(f'Similar text: {text}')
        if target != 'N/A':
            print(f'Similar target: {target}')
        print()
