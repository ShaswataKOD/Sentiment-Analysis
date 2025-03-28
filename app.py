import streamlit as st
import pickle
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure NLTK downloads in a specified directory (important for cloud environments)
nltk_data_path = "/opt/render/nltk_data"
nltk.data.path.append(nltk_data_path)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=nltk_data_path)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", download_dir=nltk_data_path)

# Initialize NLP components
porter = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
try:
    with open('model/ln_model.pkl', 'rb') as file:
        model = pickle.load(file)

    with open('model/vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
except FileNotFoundError:
    st.error("⚠️ Model or vectorizer file not found! Make sure the correct paths are provided.")
    st.stop()

# Text preprocessing function
def preprocess_and_vectorize(text):
    text = text.lower()
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'[@#]\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [porter.stem(word) for word in tokens if word not in stop_words]
    cleaned_text = ' '.join(tokens)
    vectorized_text = vectorizer.transform([cleaned_text])
    return vectorized_text, tokens

# Streamlit UI
st.set_page_config(page_title="Sentiment Analysis App", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #FF6347;'>✨ Sentiment Analysis App ✨</h1>
    <h3 style='text-align: center; color: #4682B4;'>Analyze text sentiment in real-time & batch mode!</h3>
""", unsafe_allow_html=True)

# Real-time Sentiment Scoring
user_input = st.text_area("📝 Type your text:", "", height=150, key="input_text")

if user_input.strip():
    processed_input, tokens = preprocess_and_vectorize(user_input)
    
    try:
        prediction = model.predict(processed_input)[0]
        score = model.predict_proba(processed_input)[0]
        sentiment = "Positive" if prediction >= 0.5 else "Negative"
        st.markdown(f"**Sentiment:** {sentiment} ({max(score):.2f} confidence)")

        # Keyword Highlighting (handling potential missing feature names)
        try:
            feature_names = set(vectorizer.get_feature_names_out())  # Convert to set for quick lookup
            important_words = [word for word in tokens if word in feature_names]
            highlighted_text = " ".join([
                f"<span style='color:green;font-weight:bold'>{w}</span>" if w in important_words else w for w in tokens
            ])
            st.markdown(f"**Important words:** {highlighted_text}", unsafe_allow_html=True)
        except AttributeError:
            st.warning("⚠️ Unable to retrieve feature names from vectorizer. Word highlighting disabled.")
    
    except Exception as e:
        st.error(f"⚠️ Error processing input: {e}")

# Batch Processing (File Upload)
st.markdown("---")
st.subheader("📂 Upload a file for batch sentiment analysis")
uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

if uploaded_file:
    try:
        text_data = uploaded_file.read().decode("utf-8").splitlines()
        results = []

        for line in text_data:
            if line.strip():  # Ignore blank lines
                vec_text, _ = preprocess_and_vectorize(line)
                pred = model.predict(vec_text)[0]
                conf = max(model.predict_proba(vec_text)[0])
                results.append((line, "Positive" if pred >= 0.5 else "Negative", conf))

        st.write("### Results:")
        for text, sentiment, conf in results:
            st.write(f"**{sentiment} ({conf:.2f} confidence):** {text}")

    except Exception as e:
        st.error(f"⚠️ Error processing uploaded file: {e}")

st.markdown("<div style='text-align: center;'>Made with ❤️ using Streamlit</div>", unsafe_allow_html=True)
