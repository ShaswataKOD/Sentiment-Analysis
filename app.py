import streamlit as st
import pickle
import nltk
import os
import pandas as pd  # Added since it's in requirements.txt
import numpy as np   # Added since it's in requirements.txt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Set a custom directory for NLTK data (persistent in Streamlit Cloud)
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

# Download necessary NLTK resources
try:
    nltk.download("punkt", download_dir=nltk_data_dir)
    nltk.download("stopwords", download_dir=nltk_data_dir)
except Exception as e:
    st.error(f"⚠️ Error downloading NLTK resources: {e}")
    st.stop()

# Load model and vectorizer
try:
    model = pickle.load(open("model/ln_model.pkl", "rb"))
    vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))
except FileNotFoundError:
    st.error("⚠️ Model or Vectorizer not found! Please check the files in the 'model' directory.")
    st.stop()

# Initialize NLP components
stop_words = set(stopwords.words("english"))
porter = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[@#]\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [porter.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Streamlit UI
st.set_page_config(page_title="✨ Sentiment Analyzer", layout="centered", page_icon="🔍")

# Custom CSS for Premium UI
st.markdown("""
    <style>
    body {background-color: #f5f7fa;}
    .main {background: white; padding: 40px; border-radius: 15px; box-shadow: 0px 4px 10px rgba(0,0,0,0.1);}
    .big-title {text-align: center; font-size: 45px; font-weight: bold; color: #3b5998;}
    .sub-title {text-align: center; font-size: 20px; color: #6C757D;}
    .stTextArea textarea {border-radius: 10px; border: 2px solid #3b5998; font-size: 16px; padding: 10px;}
    .sentiment-box {padding: 20px; border-radius: 10px; font-size: 18px; font-weight: bold; text-align: center; margin-top: 20px;}
    .positive {background-color: #D4EDDA; color: #155724; border: 2px solid #28a745;}
    .negative {background-color: #F8D7DA; color: #721C24; border: 2px solid #dc3545;}
    .footer {text-align: center; font-size: 14px; color: #888; margin-top: 30px;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>🔍 Sentiment Analysis App</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Analyze your text sentiment with AI-powered insights</div>", unsafe_allow_html=True)

# Text Input
st.markdown("<div class='main'>", unsafe_allow_html=True)
user_input = st.text_area("💬 Enter your text:", "", height=150)
st.markdown("</div>", unsafe_allow_html=True)

if user_input.strip():
    processed_text = preprocess(user_input)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)[0]
    sentiment = "Positive" if prediction >= 0.5 else "Negative"
    confidence = max(model.predict_proba(vectorized_text)[0])
    
    st.markdown(f"<div class='sentiment-box {'positive' if sentiment=='Positive' else 'negative'}'>
                {sentiment} Sentiment ({confidence:.2f} Confidence)
                </div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>Made with ❤️ using Streamlit</div>", unsafe_allow_html=True)
