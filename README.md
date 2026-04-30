# 🎬 Movie Recommendation System & AI Revenue Predictor

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E.svg)

A comprehensive, full-stack machine learning application that provides content-based movie recommendations using Natural Language Processing (TF-IDF) and predicts box-office revenue tiers using a Logistic Regression model. 

The project features a **FastAPI** backend that handles the machine learning models and TMDB API integrations, seamlessly connected to a modern **Streamlit** frontend dashboard.

---

## ✨ Key Features

* **🏠 Home Feed & Live TMDB Search:** Browse popular, top-rated, and upcoming movies. Live search functionality dynamically queries the TMDB API to fetch movie posters, details, and backdrops.
* **🧠 Content-Based Recommendations:** Uses NLP (TF-IDF vectorization, NLTK lemmatization, and stop-word removal) on movie metadata (overview, genres, taglines, production companies, etc.) to recommend visually and thematically similar movies.
* **📈 AI Revenue Predictor:** A trained Logistic Regression model that predicts the box-office success (Low, Medium, or High Revenue Tier) of a hypothetical movie based on user-input features (Budget, Runtime, Expected Popularity, Vote Average, and Vote Count).
* **📊 Interactive EDA Dashboard:** Explore the cinematic dataset using interactive Plotly charts, featuring dynamic filtering by budget and predicted revenue tiers.


## 📖 What is this project?

This project is a full-stack, AI-powered web application designed for movie enthusiasts, data analysts, and film producers. It serves a dual purpose:
1. **Intelligent Content Discovery:** It helps users find movies to watch through a live feed, TMDB-powered search, and a custom Natural Language Processing (NLP) recommendation engine.
2. **Box-Office Analytics & Prediction:** It allows users to explore historical movie data through an interactive dashboard and uses a trained Machine Learning model to predict the financial success tier of hypothetical movie productions.

The application is built on a decoupled architecture, utilizing **FastAPI** for high-performance, asynchronous backend operations and **Streamlit** for a reactive, modern frontend user interface.

---

## ⚙️ How It Works: The Core Mechanics

### 1. The Recommendation Engine (NLP / TF-IDF)
When a user opens a movie's details page, the app suggests "Similar Movies". 
* **The Process:** We use a **Term Frequency-Inverse Document Frequency (TF-IDF)** vectorizer. During the data preparation phase (`Movies.ipynb`), the system aggregates text data for each movie (overview, genres, keywords, cast, and crew). 
* **The Math:** It converts these massive text blocks into numerical vectors. When you request a recommendation, the backend calculates the **Cosine Similarity** between your selected movie's vector and all other movies in the database, returning the closest geometric matches.
* **Fallback:** If NLP recommendations aren't enough, it gracefully falls back to querying the TMDB API for genre-based matches.

### 2. The Revenue Predictor (Machine Learning)
The "Revenue Predictor" tab allows you to input hypothetical movie stats (Budget, Runtime, Expected Popularity, Vote Average, Vote Count) to see if it will flop or succeed.
* **The Process:** The model was trained (`ml.py`) on a cleaned historical dataset of thousands of movies. Instead of predicting exact dollar amounts (which is highly volatile), the model categorizes revenue into three distinct classes: **Low, Medium, and High Revenue**.
* **The Algorithm:** It utilizes Scikit-Learn's **Logistic Regression** (using the `saga` solver, optimized for multinomial classification).
* **The Pipeline:** User inputs are captured in Streamlit, sent as a JSON payload to FastAPI, scaled using a pre-trained `StandardScaler`, and fed into the model. The API returns the predicted tier and the model's confidence probability.

### 3. The Data Flow & API Integration
* **Asynchronous Backend:** The FastAPI backend uses `httpx.AsyncClient` to make non-blocking calls to the external TMDB API. This means if you search for "Batman", the backend rapidly fetches posters, backdrops, and metadata without freezing the server.
* **Caching Strategy:** The backend maintains an in-memory dictionary (`POSTER_CACHE`) to map TMDB IDs to image URLs. This drastically reduces redundant API calls and speeds up UI rendering for the TF-IDF recommendations.

---

## 📂 Detailed Directory & File Breakdown
```text
S:\Movie-Recommendation-System-main\movie_recommender\
│
├── data/ 
│   └── AI_movies_dataset.csv     # The raw, foundational dataset containing movie metadata.
│
├── models/                       # The "Brain" of the application.
│   ├── df.pkl                    # Serialized Pandas DataFrame containing movie metadata.
│   ├── indices.pkl               # Dictionary mapping movie titles to their DataFrame index.
│   ├── tfidf.pkl                 # The trained TF-IDF Vectorizer object.
│   ├── tfidf_matrix.pkl          # The pre-computed vector matrix for rapid similarity scoring.
│   ├── rf_classifier.pkl         # The trained Logistic Regression model for revenue prediction.
│   ├── ml_scaler.pkl             # The StandardScaler used to normalize user inputs for the ML model.
│   └── clustered_movies.pkl      # A cleaned subset of data used specifically for the Plotly EDA Dashboard.
│
├── notebooks/ 
│   └── Movies.ipynb              # Jupyter Notebook containing the data wrangling, EDA, and NLP pipeline.
│
├── .env                          # Holds environment variables securely (e.g., TMDB_API_KEY).
├── app.py                        # The Streamlit Frontend. Handles UI routing, state management, and API requests.
├── main.py                       # The FastAPI Backend. Exposes REST endpoints, loads pickle files into memory, and communicates with TMDB.
├── ml.py                         # Standalone Python script to train/retrain the Logistic Regression model and generate ml_scaler.pkl / rf_classifier.pkl.
└── requirements.txt              # Required Python libraries.
```



*(Note: Files like `test` and `AI_ML_project` were used for early testing and are excluded from the core pipeline.)*

---


## 🚀 Installation & Setup

**1. Clone the repository and navigate to the project directory:**
```bash
cd "S:\Movie-Recommendation-System-main\"
```

**2. Install dependencies:**
It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

**3. Configure Environment Variables:**
Create a `.env` file in the root directory and add your TMDB API key:
```env
TMDB_API_KEY=your_tmdb_api_key_here
```

---

## ⚙️ Data Preparation & Model Training

Before running the application, you must generate the required `.pkl` model files.

**1. Generate NLP Recommendation Models:**
Open and run all cells in `notebooks/Movies.ipynb`. This will process `data/AI_movies_dataset.csv` and export the following to the `models/` directory:
* `df.pkl`
* `indices.pkl`
* `tfidf.pkl`
* `tfidf_matrix.pkl`

**2. Train the Revenue Predictor Model:**
Run the machine learning script to train the Logistic Regression model and generate the EDA dataset.
```bash
python ml.py
```
This will output the following to the `models/` directory:
* `rf_classifier.pkl` (Logistic Regression Model)
* `ml_scaler.pkl` (Standard Scaler)
* `clustered_movies.pkl` (Cleaned DataFrame for the Dashboard)

---

## 💻 Running the Application

This project requires both the backend and frontend servers to be running simultaneously.

**1. Start the FastAPI Backend:**
Open a terminal and run:
```bash
uvicorn main:app --reload --port 8000
```
*The backend will be available at `http://127.0.0.1:8000`.*

**2. Start the Streamlit Frontend:**
Open a **new** terminal window and run:
```bash
streamlit run app.py
```
*The frontend dashboard will automatically open in your default web browser.*

---

## 🛠️ Technologies Used

* **Frontend:** Streamlit, Plotly (for interactive charts)
* **Backend:** FastAPI, Uvicorn, HTTPX (Async API calls)
* **Machine Learning & NLP:** Scikit-Learn (Logistic Regression, TF-IDF), NLTK (Text Preprocessing)
* **Data Handling:** Pandas, NumPy
* **External APIs:** TMDB API (The Movie Database)





---

*The Streamlit app will launch in your browser (usually at `http://localhost:8501`).*

---

## 🌐 API Endpoints Overview

If you wish to interact with the FastAPI backend directly, here are the core routes:

* `GET /home` - Fetches the live home feed (Popular, Top Rated, etc.) via TMDB.
* `GET /tmdb/search` - Live search endpoint for user queries.
* `GET /movie/id/{tmdb_id}` - Fetches detailed metadata, posters, and backdrops for a specific movie.
* `GET /movie/search` - The primary NLP endpoint. Returns a bundle of TF-IDF and Genre-based recommendations.
* `POST /predict/revenue` - Accepts a JSON payload of movie features and returns the predicted revenue tier and confidence intervalHere is a highly detailed and comprehensive `README.md` that dives deep into the architecture, the underlying machine learning mechanics, and the data flow of the application. You can copy and paste this directly into your repository.

---


The Streamlit app will launch in your browser (usually at http://localhost:8501).

If you wish to interact with the FastAPI backend directly, here are the core routes:

GET /home - Fetches the live home feed (Popular, Top Rated, etc.) via TMDB.

GET /tmdb/search - Live search endpoint for user queries.

GET /movie/id/{tmdb_id} - Fetches detailed metadata, posters, and backdrops for a specific movie.

GET /movie/search - The primary NLP endpoint. Returns a bundle of TF-IDF and Genre-based recommendations.

POST /predict/revenue - Accepts a JSON payload of movie features and returns the predicted revenue tier and confidence interval.

GET /api/visualization/eda - Serves the cleaned, raw dataset to the Streamlit frontend for dynamic Plotly chart rendering.

***
