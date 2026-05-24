# Movie Review Sentiment Analysis & Recommendation System 🎬🍿

This repository contains a comprehensive **Natural Language Processing (NLP)** project developed for the NLP course at **Bahria University**. The application combines **two intelligent movie-related systems in one platform**:

1. **Movie Review Sentiment Analysis** – Classifies movie reviews as **Positive** or **Negative** using machine learning.
2. **Movie Recommendation System** – Recommends similar movies using a **Content-Based Filtering** approach.

The project provides an interactive **Streamlit GUI** where users can analyze movie sentiments and get personalized movie recommendations.

---

## 📋 Project Highlights

### 🎭 1. Movie Review Sentiment Analysis
* **Balanced Dataset**: To address severe class imbalance in the initial data, the **IMDB Movie Review dataset** was integrated to provide a more robust and balanced training set.
* **Text Preprocessing**: Implements a comprehensive cleaning pipeline including:
  - HTML tag removal  
  - Punctuation stripping  
  - Lowercasing  
  - Stopword removal  
  - Tokenization
* **Feature Extraction**: Uses **TF-IDF Vectorization** to convert textual reviews into numerical representations.
* **Model Training**: Utilizes **Multinomial Naive Bayes** (or similar machine learning models) for sentiment classification.
* **Real-Time Predictions**: Users can enter custom movie reviews and instantly receive sentiment predictions (**Positive / Negative**).

---

### 🎥 2. Movie Recommendation System (Content-Based Filtering)
* **Content-Based Filtering**: Recommends movies based on similarity in movie features such as:
  - Genre
  - Keywords
  - Overview/Description
  - Cast
  - Director
* **Similarity Calculation**: Uses **Cosine Similarity** to find movies that are most similar to the selected movie.
* **Movie Search & Recommendation**: Users can search for a movie and get a list of similar recommended movies.
* **Efficient Recommendation Pipeline**: Processes textual movie metadata using **TF-IDF Vectorization** to improve recommendation quality.

---

## 🛠️ Tech Stack

### **Programming Language**
* **Python**

### **Libraries & Frameworks**
* **Data Processing**:
  - `pandas`
  - `numpy`

* **Natural Language Processing (NLP)**:
  - `nltk`
  - `scikit-learn`

* **Machine Learning & Recommendation System**:
  - `TfidfVectorizer`
  - `MultinomialNB`
  - `Cosine Similarity`

* **Visualization**:
  - `matplotlib`
  - `seaborn`

* **Deployment & Interface**:
  - `streamlit`

---

## 🚀 Features

✅ **Analyze movie review sentiment** (Positive / Negative)  
✅ **Recommend movies similar to a selected movie**  
✅ **Interactive Streamlit-based GUI**  
✅ **Real-time predictions and recommendations**  
✅ **NLP-based text preprocessing pipeline**  
✅ **Machine learning-powered classification system**

---

## 📂 Project Structure

```bash
Movie-Review-Sentiment-Recommendation-System/
│── app.py                      # Main Streamlit application
│── mnb_model.pkl               # Trained sentiment analysis model
│── mtfidf_vectorizer.pkl       # TF-IDF vectorizer
│── recommendation_model.pkl    # Recommendation system model
│──matrics.json                 # Sentiment Analysis Model Comfusion Matrics
│── movies_dataset.csv          # Movie recommendation dataset
│── requirements.txt            # Required dependencies
│── notebooks/                  # Jupyter notebooks
│── README.md                   # Project documentation
```

---

## 🚀 How to Run

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/movie-review-sentiment-analysis.git
```

### 2️⃣ Navigate to the Project Folder

```bash
cd movie-review-sentiment-analysis
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

---

## 🖥️ Application Modules

### 🎭 Sentiment Analysis Module
Users can:
- Enter a custom movie review
- Analyze review sentiment
- Get instant prediction results

### 🎥 Movie Recommendation Module
Users can:
- Search for a movie title
- Receive similar movie recommendations
- Explore movies based on content similarity

---

## 📈 Machine Learning Workflow

### **Sentiment Analysis Workflow**
1. Dataset Collection  
2. Text Preprocessing  
3. TF-IDF Feature Extraction  
4. Model Training (Naive Bayes)  
5. Sentiment Prediction  

### **Recommendation System Workflow**
1. Movie Dataset Collection  
2. Feature Combination (Genre, Cast, Director, Overview)  
3. Text Vectorization (TF-IDF)  
4. Cosine Similarity Computation  
5. Similar Movie Recommendation  

---

## 🎯 Future Improvements
* Add **collaborative filtering** recommendation system
* Improve recommendation accuracy using **deep learning**
* Deploy online using **Streamlit Cloud / Hugging Face Spaces**
* Add movie posters and trailers API integration
* Improve UI/UX for better user experience

---

## 📜 License
This project is licensed under the **MIT License** – see the `LICENSE` file for details.

---

## 👨‍💻 Developed For
**Natural Language Processing (NLP) Course**  
**Bahria University**
