
def clean_text(text): 
    
    # Remove punctuation
    cleaned_text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Lowercase all words
    cleaned_text = cleaned_text.lower()
    
    # Remove extra spaces
    cleaned_text = cleaned_text.strip()
    return cleaned_text
