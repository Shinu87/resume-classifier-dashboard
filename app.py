import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import PyPDF2
from streamlit_option_menu import option_menu
import time
import json
import google.generativeai as genai

# 🔐 Configure Gemini API
import os
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# 📥 Download required NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# 💾 Load Pretrained Models & Transformers
@st.cache_resource
def load_artifacts():
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    bert_model_name = joblib.load('bert_model_name.pkl')
    bert_model = SentenceTransformer(bert_model_name)
    logistic_model = joblib.load('hybrid_logistic_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return tfidf_vectorizer, bert_model, logistic_model, label_encoder

# 🧹 Text Preprocessing
def clean_resume(txt):
    lemmatizer = WordNetLemmatizer()
    stopword = set(stopwords.words('english'))
    txt = re.sub(r'http\S+\s', ' ', txt)
    txt = re.sub(r'@\S+', '', txt)
    txt = re.sub(r'#\S+', '', txt)
    txt = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), '', txt)
    txt = re.sub(r'\s+', ' ', txt)
    words = [lemmatizer.lemmatize(word) for word in txt.split() if word.lower() not in stopword]
    return ' '.join(words)

# 💡 Generate Gemini Suggestions
def get_resume_suggestions(resume_text, category):
    resume_text = resume_text[:4000]  # Avoid exceeding token limit
    prompt = f"""
    You are an expert resume analyzer.
    Given the resume content below and the predicted category '{category}', 
    suggest 3 specific improvements to make the resume stronger and more relevant to that category.

    Resume:
    {resume_text}
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text if hasattr(response, "text") else "⚠️ No response received."
    except Exception as e:
        return f"❌ Gemini API error: {str(e)}"

# 🎨 UI Styling
st.markdown("""
    <style>
    body {
        background-color: #f7f9fb;
        font-family: 'Segoe UI', sans-serif;
        color: #2d3436;
    }
    .main { background-color: #ffffff; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
    .header { color: #2d3436; font-weight: 700; font-size: 26px; margin-bottom: 15px; }
    .stButton > button {
        background-color: #3498db; color: #ffffff; font-weight: 600; padding: 12px 22px;
        border: none; border-radius: 10px; transition: background-color 0.3s ease;
    }
    .stButton > button:hover { background-color: #2980b9; }
    .stFileUploader, .stTextArea textarea {
        background-color: #f2f4f7; border: 1px solid #dfe6e9;
        border-radius: 10px; padding: 10px; font-size: 14px; color: #2d3436;
    }
    </style>
""", unsafe_allow_html=True)

# ⬅️ Sidebar Navigation
with st.sidebar:
    st.markdown("<h2 style='color: #ffffff;'>Resume Classifier</h2>", unsafe_allow_html=True)
    selected = option_menu(
        menu_title=None,
        options=["Dashboard", "About", "Settings"],
        icons=["house", "info-circle", "gear"],
        default_index=0,
        styles={
            "container": {"background-color": "#2c3e50"},
            "icon": {"color": "#ffffff", "font-size": "18px"},
            "nav-link": {"color": "#ffffff", "font-size": "16px", "--hover-color": "#4CAF50"},
            "nav-link-selected": {"background-color": "#4CAF50"},
        }
    )

# 🔍 Main Dashboard
tfidf_vectorizer, bert_model, logistic_model, label_encoder = load_artifacts()

if selected == "Dashboard":
    st.markdown("<div class='main'>", unsafe_allow_html=True)
    st.markdown("<div class='header'>📄 Resume Classification Dashboard</div>", unsafe_allow_html=True)
    st.write("Upload a resume in PDF format or paste text to classify and get improvement suggestions.")

    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_file = st.file_uploader("📤 Upload PDF Resume", type="pdf")
    with col2:
        resume_txt = st.text_area("📋 Paste Resume Text", height=200)

    if st.button("🔍 Classify Resume"):
        if uploaded_file or resume_txt:
            with st.spinner("Processing..."):
                time.sleep(1)

                if uploaded_file:
                    resume_txt = ""
                    reader = PyPDF2.PdfReader(uploaded_file)
                    for page in reader.pages:
                        resume_txt += page.extract_text() or ""

                if not resume_txt.strip():
                    st.error("⚠️ Could not extract text. Try another file or input.")
                    st.stop()

                cleaned = clean_resume(resume_txt)
                tfidf_vec = tfidf_vectorizer.transform([cleaned])
                bert_vec = bert_model.encode([cleaned])
                hybrid = np.hstack([tfidf_vec.toarray(), bert_vec])
                prediction = logistic_model.predict(hybrid)
                predicted_label = label_encoder.inverse_transform(prediction)[0]
                confidence = np.max(logistic_model.predict_proba(hybrid))
                probas = logistic_model.predict_proba(hybrid)[0]
                top3 = np.argsort(probas)[::-1][:3]
                top_labels = label_encoder.inverse_transform(top3)

                # 💡 Gemini Suggestions
                with st.spinner("Generating suggestions with Gemini..."):
                    suggestions = get_resume_suggestions(resume_txt, predicted_label)
                    st.markdown("### 💡 Gemini Resume Suggestions")
                    st.write(suggestions)

                st.success(f"✅ Predicted Category: **{predicted_label}**  \n🔍 Confidence: **{confidence:.2%}**")
                st.markdown("---")

                # 🔝 Top 3 Predictions
                st.markdown("### 🔝 Top 3 Predictions")
                for i in range(3):
                    st.write(f"🔹 {top_labels[i]} — {probas[top3[i]]:.2%}")

                # 🌥 Word Cloud
                st.markdown("### 🌥 Word Cloud")
                wc = WordCloud(width=800, height=400, background_color="white").generate(cleaned)
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)

                # 📥 Download
                result = {
                    "Predicted Category": predicted_label,
                    "Confidence": f"{confidence:.2%}",
                    "Top 3 Predictions": {top_labels[i]: f"{probas[top3[i]]:.2%}" for i in range(3)},
                    "Suggestions": suggestions
                }
                st.download_button("📁 Download JSON Result", data=json.dumps(result, indent=2), file_name="resume_analysis.json")

        else:
            st.warning("⚠️ Please upload a resume or paste text.")
    st.markdown("</div>", unsafe_allow_html=True)

# ℹ️ About Page
elif selected == "About":
    st.markdown("<div class='main'>", unsafe_allow_html=True)
    st.markdown("<div class='header'>ℹ️ About</div>", unsafe_allow_html=True)
    st.write("""
        This app classifies resumes into career categories using a hybrid ML model (TF-IDF + BERT + Logistic Regression).
        It also uses Gemini (Google's LLM) to suggest improvements based on the predicted category.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# ⚙️ Settings Page
elif selected == "Settings":
    st.markdown("<div class='main'>", unsafe_allow_html=True)
    st.markdown("<div class='header'>⚙️ Settings</div>", unsafe_allow_html=True)
    st.write("Future support: Upload custom models, set thresholds, and personalize suggestions.")
    st.slider("Confidence Threshold", 0.0, 1.0, 0.5, disabled=True)
    st.markdown("</div>", unsafe_allow_html=True)
