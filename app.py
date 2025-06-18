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

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

st.markdown("""
    <style>
    body {
        background-color: #f7f9fb;
        font-family: 'Segoe UI', sans-serif;
        color: #2d3436;
    }
    .main {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        color: #2d3436;
    }
    .header {
        color: #2d3436;
        font-weight: 700;
        font-size: 26px;
        margin-bottom: 15px;
    }
    .stButton > button {
        background-color: #3498db;
        color: #ffffff;
        font-weight: 600;
        padding: 12px 22px;
        border: none;
        border-radius: 10px;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #2980b9;
    }
    .stFileUploader, .stTextArea textarea {
        background-color: #f2f4f7;
        border: 1px solid #dfe6e9;
        border-radius: 10px;
        padding: 10px;
        font-size: 14px;
        color: #2d3436;
    }
    .metric-card {
        background-color: #fdfefe;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-top: 10px;
        margin-bottom: 20px;
        color: #2d3436;
    }
    .stMetric {
        background-color: #f0f3f5;
        padding: 12px;
        border-radius: 10px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        color: #2d3436;
    }
    .stAlert {
        background-color: #fef5e7 !important;
        color: #7f8c8d !important;
    }
    .sidebar .sidebar-content {
        background-color: #ecf0f1;
        color: #2d3436;
    }
    .css-1aumxhk {
        color: #2d3436 !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    bert_model_name = joblib.load('bert_model_name.pkl')
    bert_model = SentenceTransformer(bert_model_name)
    logistic_model = joblib.load('hybrid_logistic_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return tfidf_vectorizer, bert_model, logistic_model, label_encoder

def cleanResume(txt):
    lemmatizer = WordNetLemmatizer()
    stopword = set(stopwords.words('english'))
    cleanTxt = re.sub(r'http\S+\s', ' ', txt)
    cleanTxt = re.sub(r'@\S+', '', cleanTxt)
    cleanTxt = re.sub(r'#\S+', '', cleanTxt)
    cleanTxt = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), '', cleanTxt)
    cleanTxt = re.sub(r'\s+', ' ', cleanTxt)
    words = cleanTxt.split()
    words = [lemmatizer.lemmatize(word) for word in words if word.lower() not in stopword]
    return ' '.join(words)

tfidf_vectorizer, bert_model, logistic_model, label_encoder = load_artifacts()

with st.sidebar:
    st.markdown("<h2 style='color: #ffffff;'>Resume Classifier</h2>", unsafe_allow_html=True)
    selected = option_menu(
        menu_title=None,
        options=["Dashboard", "About", "Settings"],
        icons=["house", "info-circle", "gear"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"background-color": "#2c3e50"},
            "icon": {"color": "#ffffff", "font-size": "18px"},
            "nav-link": {"color": "#ffffff", "font-size": "16px", "--hover-color": "#4CAF50"},
            "nav-link-selected": {"background-color": "#4CAF50"},
        }
    )

if selected == "Dashboard":
    st.markdown("<div class='main'>", unsafe_allow_html=True)
    st.markdown("<div class='header'>📄 Resume Classification Dashboard</div>", unsafe_allow_html=True)
    st.write("Upload a resume in PDF format or paste text to classify its category using advanced ML models.")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("<div class='header'>📤 Upload Resume</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="file_uploader")
    with col2:
        st.markdown("<div class='header'>📋 Paste Resume Text</div>", unsafe_allow_html=True)
        resume_txt = st.text_area("", height=200, key="text_input")

    if st.button('🔍 Classify Resume', key="classify_button"):
        if resume_txt or uploaded_file:
            with st.spinner("Processing resume..."):
                time.sleep(1)
                if uploaded_file:
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    resume_txt = ''
                    for page in pdf_reader.pages:
                        resume_txt += page.extract_text()
                if not resume_txt.strip():
                    st.error("⚠️ Could not extract text from PDF. Please try another file.")
                    st.stop()
                cleaned_resume = cleanResume(resume_txt)
                tfidf_vectorized = tfidf_vectorizer.transform([cleaned_resume])
                bert_embeddings = bert_model.encode([cleaned_resume])
                combined_features = np.hstack((tfidf_vectorized.toarray(), bert_embeddings))
                prediction = logistic_model.predict(combined_features)
                predicted_label = label_encoder.inverse_transform(prediction)[0]
                confidence = np.max(logistic_model.predict_proba(combined_features))
                probas = logistic_model.predict_proba(combined_features)[0]
                top_n = np.argsort(probas)[::-1][:3]
                top_labels = label_encoder.inverse_transform(top_n)

                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown(f"""
    <div style="background-color: #d5f5e3; padding: 15px 20px; border-left: 6px solid #27ae60;
                border-radius: 8px; font-size: 18px; color: #000000; margin-top: 10px;">
        ✅ The resume is classified as: <strong>{predicted_label}</strong><br>
        🔍 Confidence: <strong>{confidence:.2%}</strong>
    </div>
""", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("<div class='header'>🔝 Top Predictions</div>", unsafe_allow_html=True)
                for i in range(3):
                    st.write(f"🔹 {top_labels[i]} — {probas[top_n[i]]:.2%}")

                st.markdown("<div class='header'>🌥 Word Cloud of Cleaned Resume</div>", unsafe_allow_html=True)
                wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(cleaned_resume)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)

                st.markdown("<div class='header'>📊 Classification Metrics</div>", unsafe_allow_html=True)
                metrics_html = f"""
                <div style="display: flex; gap: 20px; margin-top: 10px;">
                    <div style="flex: 1; background-color: #ffffff; padding: 20px; border-radius: 10px;
                                border-left: 5px solid #3498db; color: #000000; box-shadow: 0 2px 6px rgba(0,0,0,0.05);">
                        <h5 style="margin: 0 0 5px;">Text Length</h5>
                        <p style="font-size: 22px; font-weight: bold; margin: 0;">{len(resume_txt)} Characters</p>
                    </div>
                    <div style="flex: 1; background-color: #ffffff; padding: 20px; border-radius: 10px;
                                border-left: 5px solid #3498db; color: #000000; box-shadow: 0 2px 6px rgba(0,0,0,0.05);">
                        <h5 style="margin: 0 0 5px;">Cleaned Words</h5>
                        <p style="font-size: 22px; font-weight: bold; margin: 0;">{len(cleaned_resume.split())} Words</p>
                    </div>
                    <div style="flex: 1; background-color: #ffffff; padding: 20px; border-radius: 10px;
                                border-left: 5px solid #3498db; color: #000000; box-shadow: 0 2px 6px rgba(0,0,0,0.05);">
                        <h5 style="margin: 0 0 5px;">Processing Time</h5>
                        <p style="font-size: 22px; font-weight: bold; margin: 0;">2.5s</p>
                    </div>
                </div>
                """
                st.markdown(metrics_html, unsafe_allow_html=True)

                st.markdown("<div class='header'>📁 Download Result</div>", unsafe_allow_html=True)
                result = {
                    "Predicted Category": predicted_label,
                    "Confidence": f"{confidence:.2%}",
                    "Top 3 Predictions": {
                        top_labels[i]: f"{probas[top_n[i]]:.2%}" for i in range(3)
                    },
                    "Text Length": len(resume_txt),
                    "Cleaned Word Count": len(cleaned_resume.split())
                }
                st.download_button("📥 Download JSON", data=json.dumps(result, indent=2), file_name="classification_result.json")

                with st.expander("📃 View Raw Resume Text"):
                    st.text(resume_txt)

        else:
            st.warning("⚠️ Please upload a file or paste some resume text.")
    st.markdown("</div>", unsafe_allow_html=True)

elif selected == "About":
    st.markdown("<div class='main'>", unsafe_allow_html=True)
    st.markdown("<div class='header'>ℹ️ About the Resume Classifier</div>", unsafe_allow_html=True)
    st.write("""
        This application uses a hybrid machine learning model combining TF-IDF and BERT embeddings to classify resumes into job categories.
        Key features include:
        - 📄 PDF resume parsing
        - 🧹 Text cleaning with NLTK
        - 🌥 Interactive word cloud visualization
        - 🔍 Accurate classification with logistic regression
        - 📥 Downloadable JSON results
    """)
    st.markdown("</div>", unsafe_allow_html=True)

elif selected == "Settings":
    st.markdown("<div class='main'>", unsafe_allow_html=True)
    st.markdown("<div class='header'>⚙️ Settings</div>", unsafe_allow_html=True)
    st.write("Adjust model parameters or upload new models (future feature).")
    st.slider("Confidence Threshold", 0.0, 1.0, 0.5, disabled=True)
    st.markdown("</div>", unsafe_allow_html=True)
