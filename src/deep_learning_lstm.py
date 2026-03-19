import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def train_lstm():
    # 1. Load Data
    df = pd.read_csv("data/processed/cleaned_complaints.csv")
    X = df['cleaned_message'].astype(str).values
    y = pd.get_dummies(df['label']).values # One-hot encoding for 0, 1, 2
    
    # 2. Tokenization & Padding
    max_words = 5000
    max_len = 50 # Max length of a complaint
    
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    X_padded = pad_sequences(sequences, maxlen=max_len)
    
    # Save tokenizer for the app
    joblib.dump(tokenizer, 'models/lstm_tokenizer.pkl')
    
    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)
    
    # 4. Build LSTM Model
    model = Sequential([
        Embedding(max_words, 128, input_length=max_len),
        LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(3, activation='softmax') # 3 classes: Low, Medium, High
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # 5. Train
    print("\nTraining LSTM Model...")
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)
    
    # 6. Evaluate
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nLSTM Test Accuracy: {accuracy:.4f}")
    
    # 7. Save Model
    model.save("models/lstm_severity_model.h5")
    print("✅ LSTM model saved to models/lstm_severity_model.h5")

if __name__ == "__main__":
    train_lstm()