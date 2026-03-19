🎫 AI-Powered Customer Complaint Triage System
An end-to-end Natural Language Processing (NLP) pipeline designed to automatically classify customer complaints into High, Medium, and Low priority levels. This system assists support teams by providing real-time triage, confidence-based alerts, and actionable resolution protocols.

🚀 Project Overview
In a high-volume support environment, manual ticket prioritization is slow and prone to error. This project implements a Hybrid AI Approach—combining statistical Machine Learning (Random Forest/XGBoost) with Deep Learning (LSTM)—to ensure critical issues like fraud and security breaches are escalated within seconds.

Key Features
Negation-Aware NLP: Custom preprocessing pipeline that preserves linguistic context (e.g., distinguishing between "fraud" and "no fraud").

Dual-Model Architecture: Uses TF-IDF with Bigrams for statistical speed and LSTM for sequential deep learning.

Humble AI Logic: Implements a 75% Confidence Threshold to reduce false positives in high-priority alerts.

Interactive Triage Dashboard: A Streamlit-based UI that provides "Actionable Steps" and SLA recommendations for support agents.

🛠️ Tech Stack
Language: Python 3.13

NLP: SpaCy (Lemmatization), NLTK (Stopword Management)

Machine Learning: Scikit-Learn (Logistic Regression, Random Forest), XGBoost

Deep Learning: TensorFlow/Keras (LSTM)

Deployment: Streamlit

Data Handling: Pandas, NumPy, Joblib

📂 Project Structure

Complaint_Severity_Classifier/
├── app/
│ └── streamlit_app.py # Interactive Triage Dashboard
├── data/
│ ├── raw/ # Original dataset (complaints.csv)
│ └── processed/ # Cleaned, negation-preserved data
├── models/ # Saved .pkl and .h5 model binaries
├── src/
│ ├── preprocessing.py # SpaCy-based cleaning pipeline
│ ├── feature_extraction.py # TF-IDF Bigram vectorization
│ ├── model_train.py # ML Model training & comparison
│ ├── deep_learning_lstm.py # LSTM training & embedding logic
│ └── utils.py # Shared NLP utility functions
└── README.md # Project Documentation

⚙️ Installation & Setup

Clone the Repository

git clone https://github.com/SanjaiSK13/Customer-Intelligence-Severity-Engine
cd Customer-Intelligence-Severity-Engine

Install Dependencies

pip install -r requirements.txt
python -m spacy download en_core_web_sm

Train the Pipeline

python src/preprocessing.py
python src/feature_extraction.py
python src/model_train.py
python src/deep_learning_lstm.py

Launch the Dashboard

streamlit run app/streamlit_app.py

📊 Performance & Evaluation
Standard Metrics
Accuracy: ~99.2%

Precision: High (Optimized via 75% Confidence Threshold)

Recall: Robust across all three classes.

Contextual Robustness (Stress Test)
During development, the model was subjected to a "Blind Stress Test" involving complex negations and sarcasm (e.g., "I never said there was no problem").

Score: 4/6 Correct.

Key Finding: The model successfully handles direct negations (e.g., "no fraud") using Bigram TF-IDF features, effectively moving beyond simple keyword matching.

📋 Triage Protocol (Business Logic)
The system doesn't just label text; it provides a Response SLA:

🚨 High Priority (>75% Conf): Immediate escalation to Security/Fraud Task Force. 1-hour SLA.

⚠️ Medium Priority: Route to Logistics/Operations. 12-24 hour SLA.

✅ Low Priority: Log as general feedback or automated "Thank You" response.

🔮 Future Scope
Transformer Integration: Migrating to BERT (Bidirectional Encoder Representations from Transformers) to achieve 100% accuracy on complex logical structures like double negatives.

Multilingual Support: Expanding the SpaCy pipeline to handle global customer complaints.

API Deployment: Wrapping the model in a FastAPI container for integration into existing CRM systems (Zendesk/Salesforce).

Developed by Sanjai K
