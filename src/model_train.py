import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_all_models():
    # 1. Load Data
    df = pd.read_csv("data/processed/cleaned_complaints.csv")
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    X = tfidf.transform(df['cleaned_message'])
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. UPDATED: Using 'balanced' weights to handle the "Keyword Bias"
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
        "XGBoost": XGBClassifier(eval_metric='mlogloss')
    }
    
    best_acc = 0
    best_model = None

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model = model

    # 3. Save the Winner
    joblib.dump(best_model, 'models/best_classifier.pkl')
    print(f"\n🏆 Saved Best Model with {best_acc:.4f} accuracy.")

if __name__ == "__main__":
    train_all_models()