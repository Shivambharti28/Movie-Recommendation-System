# Movie Recommendation System

This project combines a Streamlit frontend with a FastAPI backend to recommend similar movies. The interface lets users browse TMDB-powered movie cards, open details pages, and view two recommendation tracks:

- TF-IDF recommendations from a local content-based model
- Genre recommendations fetched live from TMDB

## Architecture

- `app.py`: Streamlit UI for search, details, and recommendation grids
- `main.py`: FastAPI backend for TMDB lookups and local TF-IDF recommendations
- `build_artifacts.py`: reproducible pipeline that rebuilds the local pickle artifacts
- `Movies.ipynb`: notebook walkthrough of the dataset, feature engineering, TF-IDF model, and sanity check

## Recommendation Features

The TF-IDF model does not rely on movie overviews alone. It builds a weighted `tags` field from:

- `overview`
- `genres`
- `cast`
- `director`
- `keywords`

Before fitting TF-IDF, the text is lowercased, punctuation is removed, English stop words are removed, and each token is lemmatized.

## Dataset Source

The local recommendation artifacts are rebuilt from [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) by Rounak Banik on Kaggle.

Required raw files:

- `movies_metadata.csv`
- `credits.csv`
- `keywords.csv`

Place those files in `data/raw/` before running the artifact build step.

## Setup

1. Create and activate a virtual environment.

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Add your TMDB API key to a local `.env` file.

```env
TMDB_API_KEY=your_api_key_here
```

4. Put the raw dataset files in `data/raw/`.

5. Rebuild the local recommendation artifacts.

```bash
python build_artifacts.py
```

## Run the Project

Start the FastAPI backend:

```bash
uvicorn main:app --reload
```

In a second terminal, start the Streamlit frontend:

```bash
streamlit run app.py
```

Then open:

- FastAPI docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Streamlit app: [http://127.0.0.1:8501](http://127.0.0.1:8501)

## Sanity Check

The repo includes a simple regression test to make sure the content model still recommends Christopher Nolan neighbors for `Inception`.

Run it with:

```bash
pytest tests/test_recommendations.py -q
```

Expected behavior:

- `The Dark Knight` appears in the top 10 recommendations
- `Interstellar` appears in the top 10 recommendations

## Main API Endpoints

- `/health`: health check
- `/home`: TMDB home feed cards
- `/tmdb/search`: TMDB search results for the Streamlit search UI
- `/movie/id/{tmdb_id}`: movie details from TMDB
- `/recommend/genre`: genre-based recommendations from TMDB
- `/recommend/tfidf`: local TF-IDF recommendations by title
- `/movie/search`: bundled details plus local TF-IDF and genre recommendations

## Notes

- `.env`, `data/raw/`, local virtual environments, and `nltk_data/` are gitignored on purpose.
- The included pickle files are generated artifacts, so the model can be reproduced from scratch instead of treated as opaque binaries.
