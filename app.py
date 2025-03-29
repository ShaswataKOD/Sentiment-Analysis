import os
from pathlib import Path
import streamlit as st
import pickle
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Define a temporary NLTK data path for Render
temp_nltk_path = Path("/tmp/nltk_data")  # /tmp is writable on Render
nltk.data.path.append(str(temp_nltk_path))

# Download necessary NLTK resources if not present
nltk.download("punkt", download_dir=str(temp_nltk_path))
nltk.download("stopwords", download_dir=str(temp_nltk_path))

# Initialize NLP components
porter = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
base_dir = Path(os.getcwd()) / "model"  # Use absolute path for compatibility
model_path = base_dir / "ln_model.pkl"
vectorizer_path = base_dir / "vectorizer.pkl"

if not model_path.exists() or not vectorizer_path.exists():
    st.error("⚠️ Model or vectorizer file not found! Check the correct paths.")
    st.stop()

with open(model_path, 'rb') as file:
    model = pickle.load(file)

with open(vectorizer_path, 'rb') as file:
    vectorizer = pickle.load(file)

# Text preprocessing function
def preprocess_and_vectorize(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[@#]\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    tokens = word_tokenize(text)
    tokens = [porter.stem(word) for word in tokens if word not in stop_words]
    cleaned_text = ' '.join(tokens)
    
    if hasattr(vectorizer, 'transform'):
        vectorized_text = vectorizer.transform([cleaned_text])
    else:
        st.error("⚠️ Vectorizer is not properly loaded!")
        st.stop()
    
    return vectorized_text, tokens

# Streamlit UI
st.set_page_config(page_title="Sentiment Analysis App", layout="wide")

st.title("✨ Sentiment Analysis App ✨")
st.subheader("Analyze text sentiment in real-time & batch mode!")

# Real-time Sentiment Scoring
user_input = st.text_area("📝 Type your text:", "", height=150, key="input_text")

if user_input.strip():
    processed_input, tokens = preprocess_and_vectorize(user_input)
    
    try:
        prediction = model.predict(processed_input)[0]
        score = model.predict_proba(processed_input)[0]
        sentiment = "Positive" if prediction >= 0.5 else "Negative"
        st.write(f"**Sentiment:** {sentiment} ({max(score):.2f} confidence)")

        # Keyword Highlighting (handling potential missing feature names)
        try:
            feature_names = set(vectorizer.get_feature_names_out())
        except AttributeError:
            feature_names = set(vectorizer.get_feature_names())  # Fallback for older sklearn

        important_words = [word for word in tokens if word in feature_names]
        highlighted_text = " ".join([
            f"<span style='color:green;font-weight:bold'>{w}</span>" if w in important_words else w for w in tokens
        ])
        st.markdown(f"**Important words:** {highlighted_text}", unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"⚠️ Error processing input: {e}")

# Batch Processing (File Upload)
st.markdown("---")
st.subheader("📂 Upload a file for batch sentiment analysis")
uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

if uploaded_file:
    try:
        file_text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        lines = file_text.split("\n")
        results = []
        
        for text in lines:
            if text.strip():  # Ignore blank lines
                vec_text, _ = preprocess_and_vectorize(text)
                pred = model.predict(vec_text)[0]
                conf = max(model.predict_proba(vec_text)[0])
                results.append((text, "Positive" if pred >= 0.5 else "Negative", conf))

        st.write("### Results:")
        for text, sentiment, conf in results:
            st.write(f"**{sentiment} ({conf:.2f} confidence):** {text}")

    except Exception as e:
        st.error(f"⚠️ Error processing uploaded file: {e}")

st.markdown("<div style='text-align: center;'>Made with ❤️ using Streamlit</div>", unsafe_allow_html=True)
