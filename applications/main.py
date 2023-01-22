from rake_nltk import Rake
import spacy
# Load the spacy model for NER
nlp = spacy.load("en_core_web_sm")

text = "Barack Obama was born in Honolulu, Hawaii. He is the 44th President of the United States."

# Initialize the Rake object
r = Rake()

# Extract keywords using RAKE
r.extract_keywords_from_text(text)
keywords = r.get_ranked_phrases()
print(keywords)

# Use spaCy's NER to identify entities in text
doc = nlp(text)
for ent in doc.ents:
    print(ent.text, ent.label_)
