import os
import pandas as pd
import re
import string
import spacy
import nltk
from nltk.corpus import stopwords

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

negation_words = {'no', 'not', 'never', 'none', 'neither', 'nor'}
stop_words = set(stopwords.words('english')) - negation_words

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    # Remove punctuation except spaces
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    
    doc = nlp(text)
    # Lemmatize but ensure negation words stay
    cleaned_tokens = [
        token.lemma_ for token in doc 
        if (token.text.strip() not in stop_words or token.text.strip() in negation_words) 
        and token.is_alpha
    ]
    return " ".join(cleaned_tokens)

def preprocess_dataset(input_path, output_path):
    print(f"Reading data from: {input_path}...")
    df = pd.read_csv(input_path)
    print("Cleaning text (preserving negations)...")
    df['cleaned_message'] = df['message'].apply(clean_text)
    df = df[df['cleaned_message'] != ""]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Success! Processed data saved to: {output_path}")

if __name__ == "__main__":
    preprocess_dataset("data/raw/complaints.csv", "data/processed/cleaned_complaints.csv")