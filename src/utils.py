import re
import string
import spacy
from nltk.corpus import stopwords
import joblib

# Load SpaCy once
try:
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
except OSError:
    nlp = None

# THE KEY: Keep negation words so the model sees "not_stolen" instead of just "stolen"
negation_words = {'no', 'not', 'never', 'none', 'neither', 'nor'}
stop_words = set(stopwords.words('english')) - negation_words

def clean_single_message(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""

    text = text.lower()
    # Remove punctuation but keep spaces
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    
    if nlp:
        doc = nlp(text)
        # Keep the word if it's NOT a stopword OR if it IS a negation word
        cleaned_tokens = [
            token.lemma_ for token in doc 
            if (token.text.strip() not in stop_words or token.text.strip() in negation_words) 
            and token.is_alpha
        ]
        return " ".join(cleaned_tokens)
    else:
        # Simple fallback
        return " ".join([w for w in text.split() if w not in stop_words or w in negation_words])