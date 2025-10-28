import streamlit as st
import numpy as np
import pandas as pd
import joblib
import re
import string
import os

# -------------------------#
# 1️⃣ Safety Check for Model Files
# -------------------------#
required_files = ['rare_disease_model.pkl', 'tfidf_vectorizer.pkl', 'label_encoder.pkl']
missing = [f for f in required_files if not os.path.exists(f)]
if missing:
    st.error(f"❌ Missing model files: {', '.join(missing)}. Please upload them to run the app.")
    st.stop()

# -------------------------#
# 2️⃣ Load Model & Vectorizer
# -------------------------#
model = joblib.load('rare_disease_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# -------------------------#
# 3️⃣ Clean Text Function (NLP Preprocessing)
# -------------------------#
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -------------------------#
# 4️⃣ Page Setup & Custom Styling
# -------------------------#
st.set_page_config(page_title="🧬 Rare Disease Predictor", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg, #a0c4ff, #caffbf);
            font-family: 'Segoe UI', sans-serif;
        }
        .result-card {
            background-color: #ffffffdd;
            padding: 1rem;
            border-radius: 1rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            text-align: center;
            margin-top: 1rem;
        }
        .title-text {
            font-size: 2.2rem !important;
            color: #2b2d42;
        }
        .footer {
            text-align: center;
            color: gray;
            margin-top: 3rem;
            font-size: 0.9rem;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title-text'>🧬 Rare Disease Predictor</h1>", unsafe_allow_html=True)
st.write("Predict rare diseases from symptoms — use text or symptom selection for convenience.")

# -------------------------#
# 5️⃣ Symptom Autocomplete / Suggestions
# -------------------------#
symptom_options = [
    "fatigue", "weakness", "weight loss", "jaundice", "tremors", "speech issues",
    "memory loss", "confusion", "vision problems", "muscle weakness",
    "difficulty swallowing", "rashes", "joint pain", "fever", "abdominal pain",
    "anemia", "muscle spasms", "stiffness", "shortness of breath", "blue lips"
]

selected_symptoms = st.multiselect("🩺 Select common symptoms:", symptom_options)
user_text = st.text_area("💬 Or describe symptoms naturally:",
                         placeholder="Example: Patient has persistent fever and joint pain for 2 weeks.")

# -------------------------#
# 6️⃣ Prediction Function
# -------------------------#
def predict_disease(symptom_text):
    try:
        cleaned = clean_text(symptom_text)
        if not cleaned:
            return None, None
        X = vectorizer.transform([cleaned])
        probs = model.predict_proba(X)[0]
        top3_idx = np.argsort(probs)[::-1][:3]
        top3_diseases = label_encoder.inverse_transform(top3_idx)
        top3_probs = probs[top3_idx] * 100
        return top3_diseases, top3_probs
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
        return None, None

# -------------------------#
# 7️⃣ Predict Button
# -------------------------#
if st.button("🔍 Predict Disease"):
    combined_input = " ".join(selected_symptoms) + " " + user_text.strip()
    if not combined_input.strip():
        st.warning("⚠️ Please enter or select at least one symptom.")
    else:
        top3_diseases, top3_probs = predict_disease(combined_input)
        if top3_diseases is not None:
            main_pred, confidence = top3_diseases[0], top3_probs[0]

            # Show result
            st.markdown(f"""
            <div class='result-card'>
                <h2>🏥 Predicted Disease: <b>{main_pred}</b></h2>
                <p>Confidence: <b>{confidence:.2f}%</b></p>
            </div>
            """, unsafe_allow_html=True)

            # Top 3 predictions as side-by-side progress bars
            st.subheader("📊 Confidence Levels (Top 3 Predictions)")
            col1, col2, col3 = st.columns(3)
            cols = [col1, col2, col3]
            for i, (disease, prob) in enumerate(zip(top3_diseases, top3_probs)):
                with cols[i]:
                    st.markdown(f"**{disease}**")
                    st.progress(float(prob) / 100)
                    st.write(f"{prob:.2f}%")
        else:
            st.warning("⚠️ Unable to make prediction. Please check model or input.")

# -------------------------#
# 8️⃣ Quick Testing Section
# -------------------------#
st.markdown("---")
st.subheader("🧪 Test on Example Patient Cases")

sample_inputs = [
    "Patient has fatigue, weakness, and weight loss for the last few months.",
    "Shows yellow eyes, tremors, and trouble speaking clearly.",
    "Complains of memory loss, confusion, and blurry vision.",
    "Has trouble swallowing and muscles are getting weaker over time.",
    "Developed rashes, joint pain, and fever.",
    "Often has abdominal pain, anemia, and feels constantly tired.",
    "Experiencing muscle spasms and stiffness all over the body.",
    "Has shortness of breath, fatigue, and blue lips."
]

results = []
for text in sample_inputs:
    top3_diseases, top3_probs = predict_disease(text)
    if top3_diseases is not None:
        results.append({
            "🧾 Input Text": text,
            "🏥 Predicted Disease": top3_diseases[0],
            "🎯 Confidence (%)": round(top3_probs[0], 2),
            "2️⃣ Next Likely": top3_diseases[1],
            "3️⃣ Possible": top3_diseases[2]
        })

df = pd.DataFrame(results)

if not df.empty:
    st.dataframe(df, use_container_width=True)

# -------------------------#
# 9️⃣ Footer
# -------------------------#
st.markdown("<div class='footer'>💡 Developed as an AI-based Rare Disease Predictor — Streamlit NLP Edition</div>", unsafe_allow_html=True)
