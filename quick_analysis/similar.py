from sklearn.metrics.pairwise import cosine_similarity


def get_most_similar_texts(df, index, top_n=5):
    """
    Returns the most similar texts from a DataFrame using cosine similarity of the embeddings

    Parameters:
        df (pandas.DataFrame): DataFrame containing text embeddings
        index (int): index of the specific row in the DataFrame
        top_n (int): Number of most similar texts to return

    Returns:
        pandas.DataFrame: DataFrame containing the most similar texts, including the 'clean_text' column
    """

    # Select the row of the input text's embedding
    print(f"text to find similarity: {df['clean_text'].iloc[index]}")
    input_embedding = df.drop('clean_text', axis=1).iloc[index]
    # Calculate cosine similarity between input text and all other texts in the DataFrame
    similarity = cosine_similarity(input_embedding.values.reshape(1, -1), df.drop('clean_text', axis=1))
    # Get the indices of the top_n most similar texts
    top_indices = similarity.argsort()[0][-top_n - 1:-1][::-1]
    # Return the most similar texts
    similar_texts = df[['clean_text']].iloc[top_indices]
    for i, row in similar_texts.iterrows():
        print(row['clean_text'])
    return similar_texts
