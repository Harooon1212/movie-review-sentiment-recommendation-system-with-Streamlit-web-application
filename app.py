import streamlit as st
import pickle
import pandas as pd
import re
import string
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk

# -----------------------------
# NLTK Downloads
# -----------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# -----------------------------
# Helper Functions for Sentiment Analysis
# -----------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

# -----------------------------
# Load Models
# -----------------------------
@st.cache_data
def load_sentiment_models():
    with open("mnb_model.pkl", "rb") as f:
        nb_model = pickle.load(f)
    with open("mtfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return nb_model, vectorizer

nb_model, vectorizer = load_sentiment_models()

@st.cache_data
def load_metrics():
    with open("metrics.json") as f:
        metrics = json.load(f)
    return metrics

metrics = load_metrics()

# -----------------------------
# Load Recommender Model
# -----------------------------
df = pickle.load(open('movies.pkl', 'rb'))
cosine_sim = pickle.load(open('similarity.pkl', 'rb'))

def get_title_from_index(index):
    return df.loc[index, 'title']

def get_index_from_title(title):
    result = df[df['title'].str.lower() == title.lower()]
    if not result.empty:
        return result.index[0]
    else:
        return None

def recommend(movie_name, top_n):
    index = get_index_from_title(movie_name)
    if index is None:
        return None
    sim_scores = list(enumerate(cosine_sim[index]))
    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    recommended_movies = []
    for i, score in sorted_scores:
        recommended_movies.append((get_title_from_index(i), round(score, 3)))
    return recommended_movies

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Movie App", layout="wide")
st.title("🎬 Movie Application - Sentiment Analysis & Recommendation")

tab1, tab2, tab3 = st.tabs(["Movie Review Sentiment Analysis", "Movie Recommendation", "About"])

# -----------------------------
# TAB 1: Sentiment Analysis
# -----------------------------
with tab1:
    st.header("Movie Review Sentiment Analysis")
    st.write("Enter a movie review to predict whether it is Positive or Negative.")

    review_text = st.text_area("Enter your review:", placeholder="e.g., The movie was fantastic with amazing performances!")

    if st.button("Predict Sentiment", key="sentiment"):
        if not review_text.strip():
            st.warning("Please enter some text!")
        else:
            clean_review = preprocess_text(review_text)
            vect_review = vectorizer.transform([clean_review])
            prediction = nb_model.predict(vect_review)[0]
            sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
            color = "#2ecc71" if prediction == 1 else "#ff4b4b"

            if hasattr(nb_model, "predict_proba"):
                prob = nb_model.predict_proba(vect_review)[0]
                pos_prob = prob[1]
                neg_prob = prob[0]
            else:
                pos_prob = 0.8 if prediction == 1 else 0.2
                neg_prob = 1 - pos_prob

            col1, col2 = st.columns([2, 3])
            with col1:
                st.markdown(f"**Predicted Sentiment:** <span style='color:{color}; font-size:20px'>{sentiment}</span>", unsafe_allow_html=True)
            with col2:
                st.write("Prediction Probabilities:")
                st.progress(int(pos_prob * 100))
                st.write(f"Positive: {pos_prob:.2%}")
                st.progress(int(neg_prob * 100))
                st.write(f"Negative: {neg_prob:.2%}")

    st.markdown("---")
    st.subheader("Model Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
    col2.metric("Precision", f"{metrics['precision']*100:.2f}%")
    col3.metric("Recall", f"{metrics['recall']*100:.2f}%")
    col4.metric("F1 Score", f"{metrics['f1']*100:.2f}%")

    st.write("Confusion Matrix:")
    cm_df = pd.DataFrame(
        metrics['confusion_matrix'],
        index=["Actual Negative", "Actual Positive"],
        columns=["Predicted Negative", "Predicted Positive"]
    )
    st.dataframe(cm_df.style.background_gradient(cmap="Reds"))

# -----------------------------
# TAB 2: Content-Based Recommendation
# -----------------------------
with tab2:
    st.header("Content-Based Movie Recommender")
    st.write("Enter a movie you like and select the number of recommendations:")

    movie_input = st.text_input("Enter Movie Name:", placeholder="Type a movie name here...")
    num_recommend = st.number_input(
        "Number of Recommendations:",
        min_value=1,
        max_value=50,
        value=5,
        step=1
    )

    if st.button("Recommend Movies", key="recommend"):
        if not movie_input.strip():
            st.warning("Please enter a movie name.")
        else:
            recommendations = recommend(movie_input, num_recommend)
            if recommendations is None:
                st.error("Movie not found. Please check spelling or try another movie.")
            else:
                st.success(f"Top {num_recommend} recommendations for '{movie_input}':")
                # Create dataframe
                rec_df = pd.DataFrame(recommendations, columns=['Movie', 'Cosine Similarity'])
                rec_df.insert(0, 'No', range(1, len(rec_df)+1))
                st.table(rec_df)

# -----------------------------
# TAB 3: About
# -----------------------------
with tab3:
    st.header("About This Project")
    st.markdown("""
This project is a **Movie Application** that combines two functionalities:

1. **Movie Review Sentiment Analysis**
   - Allows users to enter a movie review.
   - Predicts whether the review is **Positive** or **Negative**.
   - Shows prediction probabilities and model performance metrics.

2. **Content-Based Movie Recommender**
   - Allows users to input a movie they like.
   - Provides top N similar movies based on content features like **genres, cast, director, and keywords**.
   - Displays recommendations in a numbered table along with **Cosine Similarity** scores.

This application is built using **Streamlit**, Python, and machine learning models.
It provides an easy-to-use interface for both predicting sentiments and finding similar movies.
""")









