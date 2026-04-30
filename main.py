import os
import pickle
import traceback
from typing import Optional, List, Dict, Any, Tuple
import re
import asyncio

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

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
app = FastAPI(title="Movie Recommender & Revenue Predictor API", version="4.0")

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
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

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

# --- NEW: API Optimization Variables ---
http_client: Optional[httpx.AsyncClient] = None
POSTER_CACHE: Dict[int, str] = {} # Caches TMDB IDs to Poster URLs

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
        return int(TITLE_TO_IDX[key])
    raise HTTPException(status_code=404, detail=f"Title not found locally: '{title}'")

# =========================
# STARTUP: LOAD ALL PICKLES
# =========================

@app.on_event("shutdown")
async def shutdown_event():
    global http_client
    if http_client:
        await http_client.aclose()

@app.on_event("startup")
async def startup_event():
    global http_client
    # Create a single reusable client to prevent socket exhaustion
    http_client = httpx.AsyncClient(timeout=20, limits=httpx.Limits(max_keepalive_connections=20, max_connections=40))
    
    # Run the synchronous pickle loading in a background thread so it doesn't block
    await asyncio.to_thread(load_pickles)

def load_pickles():
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX
    global rf_classifier, ml_scaler, clustered_df

    print("\n--- STARTING UP: LOADING MODELS & DATA ---")

    # 1. Load Recommender Data
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

    # 2. Load ML Predictive Models (Logistic Regression & Scaler)
    try:
        with open(os.path.join(BASE_DIR, "rf_classifier.pkl"), "rb") as f:
            rf_classifier = pickle.load(f)
        with open(os.path.join(BASE_DIR, "ml_scaler.pkl"), "rb") as f:
            ml_scaler = pickle.load(f)
        print("✅ Revenue Prediction models loaded")
    except Exception as e: print(f"❌ Prediction models not found: {e}")

    # 3. Load EDA Data (Formerly Clustering Data)
    try:
        with open(os.path.join(BASE_DIR, "clustered_movies.pkl"), "rb") as f:
            clustered_df = pickle.load(f)
        print("✅ EDA dataset loaded")
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

# Updated Request Model for the 5-feature Logistic Regression Predictor
class RevenuePredictionRequest(BaseModel):
    budget: float
    runtime: float
    popularity: float
    vote_average: float
    vote_count: float

class RevenuePredictionResponse(BaseModel):
    tier: int
    tier_name: str
    probability: float
    message: str

# =========================
# TMDB API UTILS
# =========================
def make_img_url(path: Optional[str]) -> Optional[str]:
    return f"{TMDB_IMG_500}{path}" if path else None

async def tmdb_get(path: str, params: Dict[str, Any], retries: int = 4) -> Dict[str, Any]:
    global http_client
    q = dict(params)
    q["api_key"] = TMDB_API_KEY
    
    for attempt in range(retries):
        try:
            r = await http_client.get(f"{TMDB_BASE}{path}", params=q)
            
            # TMDB Rate Limit (429) or Server Overload (503)
            if r.status_code in (429, 503, 504):
                wait_time = (2 ** attempt) # Exponential backoff: waits 1s, 2s, 4s...
                print(f"⚠️ TMDB limit hit. Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
                continue
                
            if r.status_code != 200:
                raise HTTPException(status_code=502, detail=f"TMDB error {r.status_code}: {r.text}")
            
            return r.json()
                
        except httpx.RequestError as e:
            if attempt == retries - 1:
                raise HTTPException(status_code=502, detail=f"TMDB request error after {retries} attempts: {repr(e)}")
            await asyncio.sleep(1)
            
    raise HTTPException(status_code=502, detail="Max retries exceeded for TMDB API")

def get_valid_poster(path_val) -> Optional[str]:
    if pd.isna(path_val) or not path_val:
        return None
    path_str = str(path_val).strip()
    if path_str.lower() in ("nan", "none", "null", ""):
        return None
    if not path_str.startswith("/"):
        path_str = "/" + path_str
    return f"{TMDB_IMG_500}{path_str}"

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

async def get_fast_recommendations(movie_title: str, movie_details: TMDBMovieDetails, top_n: int = 10) -> List[TFIDFRecItem]:
    global df, tfidf_matrix, tfidf_obj, TITLE_TO_IDX

    title_key = movie_title.strip().lower()
    
    if title_key in TITLE_TO_IDX:
        idx = TITLE_TO_IDX[title_key]
        query_vec = tfidf_matrix[idx]
    else:
        overview = movie_details.overview or ""
        genres = " ".join([g["name"] for g in movie_details.genres])
        combined_text = re.sub(r'[^a-zA-Z\s]', '', f"{overview} {genres}".lower())
        query_vec = tfidf_obj.transform([combined_text])
        
    scores = (tfidf_matrix @ query_vec.T).toarray().ravel()
    order = np.argsort(-scores)

    out = []
    for i in order:
        if len(out) >= top_n:
            break
            
        row = df.iloc[int(i)]
        rec_title = str(row["title"])
        
        if rec_title.strip().lower() == title_key:
            continue
        
        tmdb_id_raw = row.get("id")
        if pd.isna(tmdb_id_raw): 
            continue
            
        tmdb_id_int = int(tmdb_id_raw)
        
        # --- THE MAGIC FIX ---
        # 1. Check our memory cache first
        poster_url = POSTER_CACHE.get(tmdb_id_int)

        # 2. ALWAYS ask the live TMDB API if it's not in the cache (Ignore the old CSV entirely!)
        if not poster_url:
            try:
                await asyncio.sleep(0.05) # Tiny delay to keep TMDB happy
                fresh_data = await tmdb_movie_details(tmdb_id_int)
                if fresh_data and fresh_data.poster_url:
                    poster_url = fresh_data.poster_url
                    POSTER_CACHE[tmdb_id_int] = poster_url
            except Exception:
                pass 

        # 3. If TMDB truly doesn't have a poster for this movie, skip it!
        if not poster_url:
            continue
        # ---------------------

        card = TMDBMovieCard(
            tmdb_id=tmdb_id_int,
            title=rec_title,
            poster_url=poster_url
        )
        
        out.append(TFIDFRecItem(title=rec_title, score=float(scores[int(i)]), tmdb=card))
        
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

    try:
        tfidf_items = await get_fast_recommendations(best["title"], details, top_n=tfidf_top_n)
    except Exception as e:
        print(f"⚠️ NLP failed for '{details.title}': {e}")
        tfidf_items = []

    genre_recs = []
    if details.genres:
        genre_id = details.genres[0]["id"]
        discover = await tmdb_get("/discover/movie", {"with_genres": genre_id, "language": "en-US", "sort_by": "popularity.desc", "page": 1})
        cards = await tmdb_cards_from_results(discover.get("results", []), limit=genre_limit)
        genre_recs = [c for c in cards if c.tmdb_id != details.tmdb_id]

    return SearchBundleResponse(query=query, movie_details=details, tfidf_recommendations=tfidf_items, genre_recommendations=genre_recs)

# =========================
# ML ROUTES
# =========================
@app.post("/predict/revenue", response_model=RevenuePredictionResponse)
async def predict_revenue(request: RevenuePredictionRequest):
    if rf_classifier is None or ml_scaler is None:
        raise HTTPException(status_code=500, detail="ML Models not loaded. Please train them first.")

    try:
        # Order MUST match the ML.py training sequence: budget, runtime, popularity, vote_average, vote_count
        input_features = np.array([[
            request.budget, 
            request.runtime, 
            request.popularity, 
            request.vote_average, 
            request.vote_count
        ]])
        scaled_features = ml_scaler.transform(input_features)

        # Predict tier and extract probabilities
        prediction = rf_classifier.predict(scaled_features)[0]
        probabilities = rf_classifier.predict_proba(scaled_features)[0]
        
        # Map integer prediction to readable tier names
        tier_map = {0: "Low Revenue", 1: "Medium Revenue", 2: "High Revenue"}
        tier_name = tier_map.get(int(prediction), "Unknown")
        
        # Get the confidence probability of the chosen tier
        prob = float(np.max(probabilities))

        return RevenuePredictionResponse(
            tier=int(prediction),
            tier_name=tier_name,
            probability=round(prob, 4),
            message=f"The AI predicts this will be a {tier_name} movie!"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/api/visualization/eda")
def get_eda_data():
    """Serves clean, un-reduced raw data for interactive EDA dashboards. (No PCA!)"""
    global clustered_df
    if clustered_df is None:
        raise HTTPException(status_code=500, detail="EDA data not loaded")
    
    try:
        # Provide real numerical columns for proper visualizations (Scatter, Histograms, etc.)
        # Limiting to 1500 to keep the payload size manageable for the frontend
        viz_data = clustered_df[['title', 'budget', 'revenue', 'runtime', 'popularity', 'vote_average', 'revenue_tier']].head(1500).copy()
        
        # Handle NaN values to prevent JSON serialization errors
        viz_data = viz_data.fillna(0)
        
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Data format mismatch. Missing expected columns: {e}")
        
    return {"data": viz_data.to_dict(orient="records")}


@app.get("/nlp/keyword_search")
async def nlp_keyword_search(query: str = Query(...), limit: int = 12):
    global df, tfidf_matrix, tfidf_obj
    
    if df is None or tfidf_matrix is None or tfidf_obj is None:
        raise HTTPException(status_code=500, detail="NLP models not loaded. Check pickles.")

    clean_query = re.sub(r'[^a-zA-Z\s]', '', query.lower())
    
    try:
        query_vec = tfidf_obj.transform([clean_query])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vectorization failed: {e}")

    scores = (tfidf_matrix @ query_vec.T).toarray().ravel()
    order = np.argsort(-scores)

    results = []
    for i in order[:limit]:
        if scores[i] > 0.0:
            try:
                row = df.iloc[i]
                title = str(row["title"])
                tmdb_id_raw = row.get("id") 
                
                poster_url = get_valid_poster(row.get("poster_path"))
                
                if not poster_url and pd.notna(tmdb_id_raw):
                    try:
                        fresh_data = await tmdb_movie_details(int(tmdb_id_raw))
                        if fresh_data and fresh_data.poster_url:
                            poster_url = fresh_data.poster_url
                    except Exception:
                        pass
                        
                results.append(TMDBMovieCard(
                    tmdb_id=int(tmdb_id_raw) if pd.notna(tmdb_id_raw) else 0,
                    title=title,
                    poster_url=poster_url, 
                ))
            except Exception:
                continue
                
    return results