import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

def perform_tfidf(df, column='cleaned_message'):
    print("Vectorizing text using TF-IDF with BIGRAMS...")
    # ngram_range=(1, 2) is the secret sauce here
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2)) 
    tfidf_matrix = tfidf.fit_transform(df[column])
    
    joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
    print("✅ TF-IDF Bigram Vectorizer saved.")
    return tfidf_matrix, tfidf

if __name__ == "__main__":
    PROCESSED_DATA = "data/processed/cleaned_complaints.csv"
    if os.path.exists(PROCESSED_DATA):
        df = pd.read_csv(PROCESSED_DATA)
        perform_tfidf(df)
        print(f"\nFeature extraction complete with context awareness!")