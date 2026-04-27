import os
import pickle
import traceback
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv


import asyncio
import httpx
from fastapi import HTTPException
from typing import Dict, Any
import re 
# =========================
# ENV & CONSTANTS
# =========================
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    raise RuntimeError("TMDB_API_KEY missing. Put it in .env as TMDB_API_KEY=xxxx")

# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="Movie Recommender & Predictor API", version="3.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# GLOBALS & PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data & Recommender Pickles
df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None
TITLE_TO_IDX: Optional[Dict[str, int]] = None

# Machine Learning Pickles
clustered_df: Optional[pd.DataFrame] = None
rf_classifier: Any = None
ml_scaler: Any = None

# =========================
# UTILS
# =========================
def _norm_title(t: str) -> str:
    return str(t).strip().lower()

def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    title_to_idx: Dict[str, int] = {}
    if isinstance(indices, dict):
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    try:
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    except Exception as e:
        print(f"Error mapping indices: {e}")
        raise RuntimeError("indices.pkl must be dict or pandas Series-like")

def get_local_idx_by_title(title: str) -> int:
    global TITLE_TO_IDX
    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500, detail="TF-IDF index map not initialized")
    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key]) # Explicit cast to native Python int
    raise HTTPException(status_code=404, detail=f"Title not found locally: '{title}'")

# =========================
# STARTUP: LOAD ALL PICKLES
# =========================
@app.on_event("startup")
def load_pickles():
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX
    global rf_classifier, ml_scaler, clustered_df

    print("\n--- STARTING UP: LOADING MODELS & DATA ---")

    # 1. Load Recommender Data (Separated to isolate errors)
    try:
        with open(os.path.join(BASE_DIR, "df.pkl"), "rb") as f:
            df = pickle.load(f)
        print("✅ df.pkl loaded")
    except Exception as e: print(f"❌ Error loading df.pkl: {e}")

    try:
        with open(os.path.join(BASE_DIR, "indices.pkl"), "rb") as f:
            indices_obj = pickle.load(f)
        TITLE_TO_IDX = build_title_to_idx_map(indices_obj)
        print("✅ indices.pkl loaded & mapped")
    except Exception as e: print(f"❌ Error loading indices.pkl: {e}")

    try:
        with open(os.path.join(BASE_DIR, "tfidf.pkl"), "rb") as f:
            tfidf_obj = pickle.load(f)
        print("✅ tfidf.pkl loaded")
    except Exception as e: print(f"⚠️ Warning loading tfidf.pkl: {e}")

    try:
        with open(os.path.join(BASE_DIR, "tfidf_matrix.pkl"), "rb") as f:
            tfidf_matrix = pickle.load(f)
        print("✅ tfidf_matrix.pkl loaded")
    except Exception as e: print(f"❌ Error loading tfidf_matrix.pkl: {e}")

    # 2. Load ML Predictive Models
    try:
        with open(os.path.join(BASE_DIR, "rf_classifier.pkl"), "rb") as f:
            rf_classifier = pickle.load(f)
        with open(os.path.join(BASE_DIR, "ml_scaler.pkl"), "rb") as f:
            ml_scaler = pickle.load(f)
        print("✅ Blockbuster Prediction models loaded")
    except Exception as e: print(f"❌ Classification models not found: {e}")

    # 3. Load Clustering Data
    try:
        with open(os.path.join(BASE_DIR, "clustered_movies.pkl"), "rb") as f:
            clustered_df = pickle.load(f)
        print("✅ Clustering dataset loaded")
    except Exception as e: print(f"❌ Could not load clustered_movies.pkl: {e}")
    
    print("------------------------------------------\n")

# =========================
# MODELS (Pydantic)
# =========================
class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None

class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = []

class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None

class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]

class HitPredictionRequest(BaseModel):
    budget: float
    runtime: float
    popularity: float

class HitPredictionResponse(BaseModel):
    is_hit: bool
    probability: float
    message: str

# =========================
# TMDB API UTILS
# =========================
def make_img_url(path: Optional[str]) -> Optional[str]:
    return f"{TMDB_IMG_500}{path}" if path else None

async def tmdb_get(path: str, params: Dict[str, Any], retries: int = 3) -> Dict[str, Any]:
    q = dict(params)
    q["api_key"] = TMDB_API_KEY
    
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.get(f"{TMDB_BASE}{path}", params=q)
                
                if r.status_code != 200:
                    raise HTTPException(status_code=502, detail=f"TMDB error {r.status_code}: {r.text}")
                
                return r.json()
                
        except httpx.RequestError as e:
            if attempt == retries - 1:
                # If we've exhausted all retries, raise the 502 error
                raise HTTPException(status_code=502, detail=f"TMDB request error after {retries} attempts: {repr(e)}")
            
            # Wait 1 second before trying again
            await asyncio.sleep(1)

async def tmdb_cards_from_results(results: List[dict], limit: int = 20) -> List[TMDBMovieCard]:
    out: List[TMDBMovieCard] = []
    for m in (results or [])[:limit]:
        out.append(TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or m.get("name") or "",
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        ))
    return out

async def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
    data = await tmdb_get(f"/movie/{movie_id}", {"language": "en-US"})
    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres", []) or [],
    )

async def tmdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    return await tmdb_get("/search/movie", {"query": query, "include_adult": "false", "language": "en-US", "page": page})

async def tmdb_search_first(query: str) -> Optional[dict]:
    data = await tmdb_search_movies(query=query, page=1)
    results = data.get("results", [])
    return results[0] if results else None

async def attach_tmdb_card_by_title(title: str) -> Optional[TMDBMovieCard]:
    try:
        m = await tmdb_search_first(title)
        if not m: return None
        return TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or title,
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        )
    except Exception:
        return None



def tfidf_recommend_dynamic(movie_details: TMDBMovieDetails, top_n: int = 10) -> List[Tuple[str, float]]:
    global df, tfidf_matrix, tfidf_obj

    if df is None or tfidf_matrix is None or tfidf_obj is None:
        raise HTTPException(status_code=500, detail="TF-IDF resources not loaded")

    # 1. Grab the plot and genres from the live TMDB search
    overview = movie_details.overview or ""
    genres = " ".join([g["name"] for g in movie_details.genres])
    
    # Clean the text exactly like you did in your Jupyter Notebook
    combined_text = f"{overview} {genres}".lower()
    combined_text = re.sub(r'[^a-zA-Z\s]', '', combined_text) 

    # 2. Transform this brand new text into an NLP vector
    try:
        query_vec = tfidf_obj.transform([combined_text])
    except Exception as e:
        print(f"Vectorization error: {e}")
        raise HTTPException(status_code=500, detail="Failed to vectorize text")

    # 3. Calculate Cosine Similarity against your entire dataset
    try:
        scores = (tfidf_matrix @ query_vec.T).toarray().ravel()
        order = np.argsort(-scores)
    except Exception as e:
        print(f"Matrix math error: {e}")
        raise HTTPException(status_code=500, detail="Matrix multiplication failed")

    out: List[Tuple[str, float]] = []
    for i in order:
        try:
            title_i = str(df.iloc[int(i)]["title"])
            
            # Don't recommend the exact same movie back
            if title_i.strip().lower() == movie_details.title.strip().lower():
                continue
                
            out.append((title_i, float(scores[int(i)])))
        except Exception:
            continue
            
        if len(out) >= top_n:
            break
            
    return out
# =========================
# ROUTES
# =========================
@app.get("/health")
def health(): return {"status": "ok"}

@app.get("/home", response_model=List[TMDBMovieCard])
async def home(category: str = Query("popular"), limit: int = Query(24, ge=1, le=50)):
    if category == "trending":
        data = await tmdb_get("/trending/movie/day", {"language": "en-US"})
        return await tmdb_cards_from_results(data.get("results", []), limit=limit)
    data = await tmdb_get(f"/movie/{category}", {"language": "en-US", "page": 1})
    return await tmdb_cards_from_results(data.get("results", []), limit=limit)

@app.get("/tmdb/search")
async def tmdb_search(query: str = Query(..., min_length=1), page: int = Query(1, ge=1, le=10)):
    return await tmdb_search_movies(query=query, page=page)

@app.get("/movie/id/{tmdb_id}", response_model=TMDBMovieDetails)
async def movie_details_route(tmdb_id: int):
    return await tmdb_movie_details(tmdb_id)

@app.get("/recommend/genre", response_model=List[TMDBMovieCard])
async def recommend_genre(tmdb_id: int = Query(...), limit: int = Query(18, ge=1, le=50)):
    details = await tmdb_movie_details(tmdb_id)
    if not details.genres: return []
    genre_id = details.genres[0]["id"]
    discover = await tmdb_get("/discover/movie", {"with_genres": genre_id, "language": "en-US", "sort_by": "popularity.desc", "page": 1})
    cards = await tmdb_cards_from_results(discover.get("results", []), limit=limit)
    return [c for c in cards if c.tmdb_id != tmdb_id]

@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(query: str = Query(..., min_length=1), tfidf_top_n: int = Query(12, ge=1, le=30), genre_limit: int = Query(12, ge=1, le=30)):
    best = await tmdb_search_first(query)
    if not best:
        raise HTTPException(status_code=404, detail=f"No TMDB movie found for query: {query}")

    tmdb_id = int(best["id"])
    details = await tmdb_movie_details(tmdb_id)

    tfidf_items: List[TFIDFRecItem] = []
    
    # Real NLP TF-IDF Dynamic Vectorization
    try:
        recs = tfidf_recommend_dynamic(details, top_n=tfidf_top_n)
    except Exception as e:
        print(f"⚠️ Dynamic NLP failed for '{details.title}': {e}")
        recs = []

    for title, score in recs:
        card = await attach_tmdb_card_by_title(title)
        tfidf_items.append(TFIDFRecItem(title=title, score=score, tmdb=card))

    genre_recs: List[TMDBMovieCard] = []
    if details.genres:
        genre_id = details.genres[0]["id"]
        discover = await tmdb_get("/discover/movie", {"with_genres": genre_id, "language": "en-US", "sort_by": "popularity.desc", "page": 1})
        cards = await tmdb_cards_from_results(discover.get("results", []), limit=genre_limit)
        genre_recs = [c for c in cards if c.tmdb_id != details.tmdb_id]

    return SearchBundleResponse(query=query, movie_details=details, tfidf_recommendations=tfidf_items, genre_recommendations=genre_recs)

# =========================
# ML ROUTES
# =========================
@app.post("/predict/hit", response_model=HitPredictionResponse)
async def predict_hit(request: HitPredictionRequest):
    if rf_classifier is None or ml_scaler is None:
        raise HTTPException(status_code=500, detail="ML Models not loaded. Please train them first.")

    try:
        input_features = np.array([[request.budget, request.runtime, request.popularity]])
        scaled_features = ml_scaler.transform(input_features)

        prediction = rf_classifier.predict(scaled_features)[0]
        probabilities = rf_classifier.predict_proba(scaled_features)[0]

        is_hit = bool(prediction == 1)
        prob = float(probabilities[1]) if is_hit else float(probabilities[0])

        return HitPredictionResponse(
            is_hit=is_hit,
            probability=round(prob, 4),
            message="Looks like a Blockbuster!" if is_hit else "Likely a Flop."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/api/visualization/clusters")
def get_cluster_data():
    global clustered_df
    if clustered_df is None:
        raise HTTPException(status_code=500, detail="Clustering data not loaded")
    
    try:
        viz_data = clustered_df[['title', 'pca_x', 'pca_y', 'cluster']].copy()
        viz_data = viz_data.rename(columns={'pca_x': 'PCA1', 'pca_y': 'PCA2', 'cluster': 'Cluster'})
    except KeyError:
        viz_data = clustered_df[['title', 'PCA1', 'PCA2', 'Cluster']]
        
    return {"clusters": viz_data.to_dict(orient="records")}




@app.get("/nlp/keyword_search")
async def nlp_keyword_search(query: str = Query(...), limit: int = 12):
    global df, tfidf_matrix, tfidf_obj
    
    if df is None or tfidf_matrix is None or tfidf_obj is None:
        raise HTTPException(status_code=500, detail="NLP models not loaded. Check pickles.")

    # 1. Clean the raw keywords
    clean_query = re.sub(r'[^a-zA-Z\s]', '', query.lower())
    
    # 2. Convert keywords to a mathematical vector using your trained model
    try:
        query_vec = tfidf_obj.transform([clean_query])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vectorization failed: {e}")

    # 3. Calculate Cosine Similarity across all 8,963 movie plots
    scores = (tfidf_matrix @ query_vec.T).toarray().ravel()
    order = np.argsort(-scores)

    results = []
    for i in order[:limit]:
        # Only return movies that actually share some semantic similarity
        if scores[i] > 0.0:
            try:
                title = str(df.iloc[i]["title"])
                tmdb_id_raw = df.iloc[i].get("id") # Grab TMDB ID straight from your CSV data!
                
                # Build the card instantly without asking TMDB for permission
                results.append(TMDBMovieCard(
                    tmdb_id=int(tmdb_id_raw) if pd.notna(tmdb_id_raw) else 0,
                    title=title,
                    poster_url=None, # We'll let the frontend handle missing posters gracefully
                ))
            except Exception:
                continue
                
    return results