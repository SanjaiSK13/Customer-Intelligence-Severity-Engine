import streamlit as st
import joblib
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import clean_single_message

# 1. Page Configuration
st.set_page_config(page_title="AI Triage Assistant", page_icon="⚖️", layout="centered")

# 2. Load the AI "Brain"
@st.cache_resource
def load_assets():
    try:
        vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
        model = joblib.load("models/best_classifier.pkl")
        return vectorizer, model
    except Exception as e:
        st.error(f"Error loading models: {e}. Please ensure you have trained the model first!")
        return None, None

vectorizer, model = load_assets()

# --- UI DESIGN ---
st.title("🎫 Customer Complaint Triage AI")
st.markdown("""
    *Automated classification and response prioritization powered by Machine Learning.*
    ---
""")

user_input = st.text_area("✍️ Enter Customer Message:", placeholder="Example: My account was hacked and I see a fraud charge...", height=150)

if st.button("Analyze & Triage"):
    if user_input.strip() and vectorizer:
        # A. Preprocess
        cleaned = clean_single_message(user_input)
        vec = vectorizer.transform([cleaned])
        
        # B. Get Probabilities (The 75% Confidence Rule)
        probs = model.predict_proba(vec)[0]
        
        # C. Logic for Actionable Steps
        if probs[2] > 0.75:
            # --- HIGH PRIORITY ---
            st.error(f"🚨 **CRITICAL PRIORITY** (Confidence: {probs[2]:.1%})")
            st.markdown("### 🛠️ Recommended Actions:")
            st.info("""
                1. **Immediate Escalation:** Notify the **Security/Fraud Task Force** immediately.
                2. **Account Lock:** Temporarily suspend account access to prevent further loss.
                3. **Customer Outreach:** Trigger an automated 'Emergency Security' call/email.
                4. **SLA:** Resolution required within **1 hour**.
            """)
            
        elif (probs[1] + probs[2]) > 0.5:
            # --- MEDIUM PRIORITY ---
            st.warning(f"⚠️ **STANDARD PRIORITY** (Confidence: {probs[1]:.1%})")
            st.markdown("### 🛠️ Recommended Actions:")
            st.info("""
                1. **Assign to Logistics:** Route this ticket to the **Shipping/Inventory Team**.
                2. **Status Check:** Verify the tracking number or system logs automatically.
                3. **SLA:** Standard response within **12-24 hours**.
            """)
            
        else:
            # --- LOW PRIORITY ---
            st.success(f"✅ **LOW PRIORITY / FEEDBACK** (Confidence: {probs[0]:.1%})")
            st.markdown("### 🛠️ Recommended Actions:")
            st.info("""
                1. **Log Feedback:** Move this to the **Product Improvement** dashboard.
                2. **Auto-Reply:** Send a 'Thank You' email with a link to the Knowledge Base.
                3. **SLA:** No immediate action required; review in weekly batch.
            """)
            
        # D. Transparency for the User
        with st.expander("🔍 See AI Reasoning"):
            st.write(f"**Cleaned Keywords identified:** `{cleaned}`")
            st.bar_chart({"Low": probs[0], "Medium": probs[1], "High": probs[2]})

    else:
        st.warning("Please enter a message to analyze.")

# --- FOOTER ---
st.markdown("---")
st.caption("Developed by Sanjai | Complaint Severity Classifier v2.0 (Negation-Aware)")