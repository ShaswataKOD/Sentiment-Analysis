import streamlit as st
import pickle
import nltk
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Set a custom directory for NLTK data (persistent in Streamlit Cloud)
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")  # This will be /app/<your-repo-name>/nltk_data
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

# Download necessary NLTK resources
try:
    nltk.download("punkt", download_dir=nltk_data_dir)
    nltk.download("punkt_tab", download_dir=nltk_data_dir)  # Required for Punkt tokenizer
    nltk.download("stopwords", download_dir=nltk_data_dir)
except Exception as e:
    st.error(f"⚠️ Error downloading NLTK resources: {e}")
    st.stop()

# Load model and vectorizer
try:
    # Ensure the path is correct relative to the root of your repo
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
st.set_page_config(page_title="Sentiment Analyzer", layout="wide")

st.markdown("""
    <style>
    .big-title {text-align: center; font-size: 40px; font-weight: bold; color: #4A90E2;}
    .sub-title {text-align: center; font-size: 20px; color: #6C757D;}
    .stTextArea {border-radius: 10px; border: 2px solid #4A90E2;}
    .sentiment-box {padding: 15px; border-radius: 10px; font-weight: bold; text-align: center;}
    .positive {background-color: #D4EDDA; color: #155724;}
    .negative {background-color: #F8D7DA; color: #721C24;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>🔍 Sentiment Analysis App</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Analyze text sentiment effortlessly</div>", unsafe_allow_html=True)

# Text Input
user_input = st.text_area("Type your text below", "", height=150)

if user_input.strip():
    processed_text = preprocess(user_input)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)[0]
    sentiment = "Positive" if prediction >= 0.5 else "Negative"
    confidence = max(model.predict_proba(vectorized_text)[0])
    
    st.markdown(f"<div class='sentiment-box {'positive' if sentiment=='Positive' else 'negative'}'>\n"
                f"{sentiment} Sentiment ({confidence:.2f} Confidence)\n" 
                "</div>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; font-size: 14px;'>Made with ❤️ using Streamlit</div>", unsafe_allow_html=True)
