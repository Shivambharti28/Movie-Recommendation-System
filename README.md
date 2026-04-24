# Moviesss AI

Moviesss AI is a streaming-style movie recommendation app powered by TMDB and a hybrid NLP recommendation engine.

## Features

- live TMDB-powered movie discovery
- smooth responsive frontend with hero banners, rails, and search
- exact movie title matching before free-text recommendation fallback
- same-language recommendations shown first, then cross-language recommendations
- industry-aware matching so Bollywood stays mostly Bollywood, Hollywood stays mostly Hollywood, and so on
- collection-style search intents such as `bollywood`, `hollywood`, `korean cinema`, and `anime movies`
- zero-build Python backend

## Setup

1. Add your TMDB key in [.env](/Users/shivambharti/Desktop/Moviesss/.env).
2. Start the app:

```bash
python3 app.py
```

3. Open `http://127.0.0.1:8000`

## Running On Another Port

If port `8000` is already in use, the app will show a friendly message instead of crashing.

To run a second instance:

```bash
PORT=8001 python3 app.py
```

## Search Behavior

The app supports three main discovery modes:

- movie title search:
  Finds the exact movie first, then recommends similar movies.
- collection search:
  Queries like `bollywood`, `hollywood`, `hindi movies`, `korean cinema`, or `anime movies` return curated live collections.
- free-text preference search:
  Queries like `emotional sci-fi survival` or `dark detective thriller` use semantic matching over the local catalog.

## Recommendation Behavior

When the app finds a movie anchor, it now orders recommendations like this:

1. same-language similar movies
2. different-language similar movies

For example:

- a Hindi movie shows Hindi recommendations first, then non-Hindi movies
- an English movie shows English recommendations first, then other languages

The recommender also applies industry-aware bias:

- Bollywood anchors prefer Bollywood titles
- Hollywood anchors prefer Hollywood titles
- other cinemas are grouped by language or regional industry where possible

## Recommendation Approach

The ranking blends several signals:

- TMDB recommendations and TMDB similar-movie graph
- TF-IDF style semantic similarity over title, overview, tagline, genres, cast, directors, and keywords
- same-language and same-industry prioritization
- genre, cast, director, and keyword overlap
- quality weighting from rating, vote volume, and popularity

## Data And Caching

- the app stores the cached TMDB catalog in `data/movies_cache.json`
- refreshing the catalog rebuilds local NLP vectors from the latest snapshot
- some recommendation flows also use live TMDB lookups for better movie-specific similarity

## Notes

- the backend is intentionally dependency-light and runs with plain `python3`
- if browser changes do not appear immediately, do a hard refresh to reload the latest frontend assets
