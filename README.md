# Movie Recommendation System

This project is a movie recommendation system built with FastAPI, HTML, CSS, and JavaScript. It lets users browse popular movies, search for titles, open movie details, and view recommendations based on TF-IDF similarity and genre matching.

## Features

- Browse home feed categories like trending, popular, top rated, now playing, and upcoming
- Search movies using TMDB
- View movie posters, release year, genres, and overview
- Get TF-IDF based similar movie recommendations
- Get genre-based recommendations
- Responsive frontend served directly by FastAPI

## Project Structure

```text
.
├── app.py
├── main.py
├── df.pkl
├── indices.pkl
├── tfidf.pkl
├── tfidf_matrix.pkl
├── requirements.txt
└── static/
    ├── index.html
    ├── styles.css
    └── script.js
```

## Requirements

- Python 3.10+
- TMDB API key

## Setup

1. Create and activate a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Add your TMDB API key in a `.env` file.

```env
TMDB_API_KEY=your_api_key_here
```

## Run the Project

Start the FastAPI server with:

```bash
uvicorn main:app --reload
```

Then open:

[http://127.0.0.1:8000](http://127.0.0.1:8000)

## Main Endpoints

- `/` - frontend
- `/health` - health check
- `/home` - home feed movies
- `/tmdb/search` - movie search from TMDB
- `/movie/id/{tmdb_id}` - movie details
- `/recommend/genre` - genre-based recommendations
- `/recommend/tfidf` - TF-IDF recommendations by title
- `/movie/search` - bundled details and recommendations

## Notes

- The recommendation model uses the local pickle files already included in the project.
- The frontend uses your FastAPI routes directly, so there is no separate frontend server required.
- If TMDB data does not load, check your API key and internet connection.
