import json
import math
import os
import re
import shlex
import shutil
import socket
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"
CACHE_FILE = DATA_DIR / "movies_cache.json"
CONFIG_FILE = BASE_DIR / ".env"

HOST = "127.0.0.1"
PORT = int(os.environ.get("PORT", "8000"))
TMDB_BASE = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p/w780"
BACKDROP_BASE = "https://image.tmdb.org/t/p/original"
CACHE_TTL_SECONDS = 60 * 60 * 8
MAX_DATASET_SIZE = 60

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "their", "there", "they", "this", "to", "was", "were", "will", "with",
    "into", "about", "after", "before", "through", "during", "while", "his",
    "her", "him", "she", "you", "your", "we", "our", "them", "who", "what",
    "when", "where", "why", "how", "up", "down", "over", "under", "again",
    "than", "then", "out", "off", "very", "can", "could", "should", "would",
    "just", "more", "most", "some", "such", "no", "not", "only", "own",
}

LANGUAGE_LABELS = {
    "ar": "Arabic",
    "bn": "Bengali",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "ko": "Korean",
    "ml": "Malayalam",
    "mr": "Marathi",
    "pt": "Portuguese",
    "ru": "Russian",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tr": "Turkish",
    "ur": "Urdu",
    "zh": "Chinese",
}


def read_env_file(path: Path) -> dict:
    values = {}
    if not path.exists():
        return values
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_api_key() -> str:
    env_values = read_env_file(CONFIG_FILE)
    return os.environ.get("TMDB_API_KEY") or env_values.get("TMDB_API_KEY", "").strip()


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9']+", (text or "").lower())
    return [word for word in words if word not in STOPWORDS and len(word) > 1]


def cosine_similarity(vec_a: dict, norm_a: float, vec_b: dict, norm_b: float) -> float:
    if not vec_a or not vec_b or not norm_a or not norm_b:
        return 0.0
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    dot = 0.0
    for token, weight in vec_a.items():
        dot += weight * vec_b.get(token, 0.0)
    return dot / (norm_a * norm_b) if dot else 0.0


class TMDBClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.curl_candidates = self._build_curl_candidates()

    def get(self, path: str, **params):
        if not self.api_key:
            raise RuntimeError("TMDB_API_KEY is missing.")
        query = {"api_key": self.api_key, "language": "en-US", **params}
        url = f"{TMDB_BASE}{path}?{urllib.parse.urlencode(query)}"
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "accept": "application/json",
                    "user-agent": "Moviesss/1.0",
                },
            )
            with urllib.request.urlopen(request, timeout=8) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            # Some local Python/OpenSSL setups fail the TMDB TLS handshake even though
            # the user's interactive shell can fetch the same URL successfully.
            # Fall back to a login-shell curl so the app still hydrates on those machines.
            return self._get_with_shell_curl(url, exc)

    def _get_with_shell_curl(self, url: str, original_error: Exception):
        errors = []
        for extra_args in [[], ["--insecure"]]:
            for curl_path in self.curl_candidates:
                result = subprocess.run(
                    [curl_path, "-sS", "--fail", *extra_args, url],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return json.loads(result.stdout)
                flag_label = " insecure" if extra_args else ""
                errors.append(f"{curl_path}{flag_label}: {(result.stderr or '').strip() or 'request failed'}")

        # Fall back to a login shell last, in case the user has a working curl setup there.
        for command in [
            f"curl -sS --fail {shlex.quote(url)}",
            f"curl -sS --fail --insecure {shlex.quote(url)}",
        ]:
            result = subprocess.run(
                ["zsh", "-lc", command],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout)
            if result.returncode != 0:
                errors.append(f"zsh login curl: {(result.stderr or '').strip() or 'request failed'}")
        raise RuntimeError("; ".join(errors) or str(original_error))

    def _build_curl_candidates(self):
        candidates = []
        for path in [shutil.which("curl"), "/opt/anaconda3/bin/curl", "/usr/bin/curl"]:
            if path and path not in candidates and Path(path).exists():
                candidates.append(path)
        return candidates

    def fetch_catalog(self):
        buckets = {
            "trending": self._safe_bucket(lambda: self.get("/trending/movie/week")),
            "bollywood": self._safe_bucket(
                lambda: {
                    "results": self.discover_movies(
                        with_original_language="hi",
                        sort_by="popularity.desc",
                        page=1,
                    )
                }
            ),
            "popular": self._safe_bucket(lambda: self.get("/movie/popular", page=1)),
            "top_rated": self._safe_bucket(lambda: self.get("/movie/top_rated", page=1)),
            "upcoming": self._safe_bucket(lambda: self.get("/movie/upcoming", page=1)),
        }
        seen = {}
        curated_ids = []
        section_queues = {
            section_name: [movie.get("id") for movie in payload.get("results", []) if movie.get("id")]
            for section_name, payload in buckets.items()
        }
        while len(curated_ids) < MAX_DATASET_SIZE and any(section_queues.values()):
            for section_name, queue in section_queues.items():
                while queue:
                    movie_id = queue.pop(0)
                    if movie_id not in seen:
                        seen[movie_id] = section_name
                        curated_ids.append(movie_id)
                        break
                if len(curated_ids) >= MAX_DATASET_SIZE:
                    break
        details = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(self.movie_details, movie_id): movie_id
                for movie_id in curated_ids[:MAX_DATASET_SIZE]
            }
            for future in as_completed(futures):
                try:
                    details.append(future.result())
                except Exception:
                    continue
        return {"fetched_at": time.time(), "sections": buckets, "details": details}

    def _safe_bucket(self, loader):
        try:
            payload = loader()
            return payload if isinstance(payload, dict) else {"results": payload}
        except Exception:
            return {"results": []}

    def search_movies(self, query: str):
        payload = self.get("/search/movie", query=query, include_adult="false", page=1)
        return payload.get("results", [])

    def discover_movies(self, **params):
        payload = self.get("/discover/movie", **params)
        return payload.get("results", [])

    def movie_details(self, movie_id: int):
        return self.get(f"/movie/{movie_id}", append_to_response="credits,keywords,videos")

    def movie_graph_bundle(self, movie_id: int):
        details = self.movie_details(movie_id)
        details["recommendations"] = self._safe_get(
            f"/movie/{movie_id}/recommendations",
            default={"results": []},
            page=1,
        )
        details["similar"] = self._safe_get(
            f"/movie/{movie_id}/similar",
            default={"results": []},
            page=1,
        )
        return details

    def _safe_get(self, path: str, default=None, **params):
        try:
            return self.get(path, **params)
        except Exception:
            return default


class RecommendationEngine:
    def __init__(self):
        self.dataset = {"movies": [], "sections": {}, "fetched_at": 0}
        self.movie_map = {}
        self.idf = {}
        self.genre_counter = Counter()
        self.lock = threading.Lock()

    def load(self, raw_payload: dict):
        sections = raw_payload.get("sections", {})
        details = raw_payload.get("details", [])
        processed_movies = []
        genre_counter = Counter()

        docs_tokens = []
        for movie in details:
            processed = self._process_movie(movie)
            if not processed:
                continue
            processed_movies.append(processed)
            docs_tokens.append(set(processed["tokens"]))
            genre_counter.update(processed["genres"])

        idf = {}
        total_docs = max(len(processed_movies), 1)
        doc_frequency = Counter()
        for tokens in docs_tokens:
            doc_frequency.update(tokens)
        for token, frequency in doc_frequency.items():
            idf[token] = math.log((1 + total_docs) / (1 + frequency)) + 1.0

        movie_map = {}
        for movie in processed_movies:
            tf = Counter(movie["tokens"])
            max_tf = max(tf.values(), default=1)
            vector = {
                token: ((0.5 + 0.5 * count / max_tf) * idf.get(token, 1.0))
                for token, count in tf.items()
            }
            norm = math.sqrt(sum(weight * weight for weight in vector.values()))
            movie["vector"] = vector
            movie["norm"] = norm
            movie_map[movie["id"]] = movie

        normalized_sections = {}
        for name, payload in sections.items():
            cards = []
            for movie in payload.get("results", [])[:18]:
                normalized = self._normalize_card(movie_map.get(movie.get("id")) or movie)
                if normalized:
                    cards.append(normalized)
            normalized_sections[name] = cards

        with self.lock:
            self.dataset = {
                "movies": processed_movies,
                "sections": normalized_sections,
                "fetched_at": raw_payload.get("fetched_at", time.time()),
            }
            self.movie_map = movie_map
            self.idf = idf
            self.genre_counter = genre_counter

    def _process_movie(self, movie: dict):
        overview = movie.get("overview") or ""
        title = movie.get("title") or ""
        tagline = movie.get("tagline") or ""
        country_codes = self._extract_country_codes(movie)
        industry_key, industry_label = self._industry_profile(movie.get("original_language") or "", country_codes)
        genres = [genre["name"] for genre in movie.get("genres", []) if genre.get("name")]
        keywords = [item["name"] for item in movie.get("keywords", {}).get("keywords", []) if item.get("name")]
        cast = [item["name"] for item in movie.get("credits", {}).get("cast", [])[:6] if item.get("name")]
        directors = [
            item["name"]
            for item in movie.get("credits", {}).get("crew", [])
            if item.get("job") == "Director" and item.get("name")
        ][:2]
        rec_ids = [item["id"] for item in movie.get("recommendations", {}).get("results", [])[:8] if item.get("id")]
        similar_ids = [item["id"] for item in movie.get("similar", {}).get("results", [])[:8] if item.get("id")]
        release_year = (movie.get("release_date") or "0000")[:4]

        corpus = " ".join(
            [
                title,
                tagline,
                overview,
                " ".join(genres),
                " ".join(keywords),
                " ".join(cast),
                " ".join(directors),
            ]
        )
        tokens = tokenize(corpus)
        if not tokens:
            return None

        return {
            "id": movie["id"],
            "title": title,
            "overview": overview,
            "tagline": tagline,
            "original_language": movie.get("original_language") or "",
            "country_codes": country_codes,
            "industry_key": industry_key,
            "industry_label": industry_label,
            "genres": genres,
            "keywords": keywords,
            "cast": cast,
            "directors": directors,
            "release_year": release_year,
            "runtime": movie.get("runtime") or 0,
            "vote_average": movie.get("vote_average") or 0.0,
            "vote_count": movie.get("vote_count") or 0,
            "popularity": movie.get("popularity") or 0.0,
            "poster_url": f"{IMAGE_BASE}{movie['poster_path']}" if movie.get("poster_path") else "",
            "backdrop_url": f"{BACKDROP_BASE}{movie['backdrop_path']}" if movie.get("backdrop_path") else "",
            "tokens": tokens,
            "recommendation_ids": rec_ids,
            "similar_ids": similar_ids,
            "trailer_key": self._extract_trailer_key(movie.get("videos", {}).get("results", [])),
        }

    def _extract_trailer_key(self, videos: list[dict]) -> str:
        for video in videos:
            if video.get("site") == "YouTube" and video.get("type") == "Trailer" and video.get("key"):
                return video["key"]
        return ""

    def _normalize_card(self, movie: dict):
        if not movie:
            return None
        original_language = movie.get("original_language", "")
        country_codes = self._extract_country_codes(movie)
        industry_key, industry_label = self._industry_profile(original_language, country_codes)
        return {
            "id": movie.get("id"),
            "title": movie.get("title", ""),
            "overview": movie.get("overview", ""),
            "poster_url": movie.get("poster_url") or (f"{IMAGE_BASE}{movie['poster_path']}" if movie.get("poster_path") else ""),
            "backdrop_url": movie.get("backdrop_url") or (f"{BACKDROP_BASE}{movie['backdrop_path']}" if movie.get("backdrop_path") else ""),
            "vote_average": round(movie.get("vote_average", 0.0), 1),
            "release_year": movie.get("release_year") or ((movie.get("release_date") or "0000")[:4]),
            "original_language": original_language,
            "country_codes": country_codes,
            "industry_key": industry_key,
            "industry_label": industry_label,
            "genres": movie.get("genres", []),
            "runtime": movie.get("runtime", 0),
            "trailer_key": movie.get("trailer_key", ""),
        }

    def search(self, query: str):
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        vector = self._build_query_vector(query_tokens)
        results = []
        with self.lock:
            for movie in self.dataset["movies"]:
                similarity = cosine_similarity(vector, self._norm(vector), movie["vector"], movie["norm"])
                title_score = self._title_match_score(query, movie["title"])
                combined_similarity = max(similarity, title_score)
                if combined_similarity <= 0:
                    continue
                score = combined_similarity * 0.7 + self._quality_score(movie) * 0.3
                results.append((score, movie))
        results.sort(key=lambda item: item[0], reverse=True)
        return [self._movie_payload(movie, score=round(score, 4)) for score, movie in results[:24]]

    def best_title_match(self, query: str):
        normalized_query = self._normalize_text(query)
        if not normalized_query:
            return None
        best_score = 0.0
        best_movie = None
        with self.lock:
            for movie in self.dataset["movies"]:
                title_score = self._title_match_score(query, movie["title"])
                if title_score > best_score:
                    best_score = title_score
                    best_movie = movie
        if best_movie and best_score >= 0.72:
            return self._movie_payload(best_movie, score=round(best_score, 4))
        return None

    def recommend(self, movie_id: int | None = None, query: str = "", genres: list[str] | None = None):
        genres = genres or []
        with self.lock:
            movies = list(self.dataset["movies"])
            anchor_movie = self.movie_map.get(movie_id) if movie_id else None
            query_vector = self._build_query_vector(tokenize(query)) if query else {}

        scores = []
        for movie in movies:
            if movie_id and movie["id"] == movie_id:
                continue
            score = 0.0
            if anchor_movie:
                score += cosine_similarity(anchor_movie["vector"], anchor_movie["norm"], movie["vector"], movie["norm"]) * 0.55
                if movie["id"] in anchor_movie["recommendation_ids"]:
                    score += 0.12
                if movie["id"] in anchor_movie["similar_ids"]:
                    score += 0.08
                overlap = len(set(anchor_movie["genres"]) & set(movie["genres"]))
                score += min(overlap * 0.03, 0.12)
            if query_vector:
                score += cosine_similarity(query_vector, self._norm(query_vector), movie["vector"], movie["norm"]) * 0.45
            if genres:
                overlap = len(set(genres) & set(movie["genres"]))
                score += overlap * 0.04
            score += self._quality_score(movie) * 0.22
            if score > 0:
                scores.append((score, movie))

        scores.sort(key=lambda item: item[0], reverse=True)
        return [self._movie_payload(movie, score=round(score, 4)) for score, movie in scores[:24]]

    def bootstrap(self):
        with self.lock:
            hero = max(
                self.dataset["movies"],
                key=lambda movie: (movie["popularity"] * 0.6 + movie["vote_average"] * 15),
                default=None,
            )
            genres = [genre for genre, _ in self.genre_counter.most_common(8)]
            return {
                "hero": self._movie_payload(hero) if hero else None,
                "sections": self.dataset["sections"],
                "genres": genres,
                "stats": {
                    "catalog_size": len(self.dataset["movies"]),
                    "last_refresh": self.dataset["fetched_at"],
                },
            }

    def _movie_payload(self, movie: dict | None, score: float | None = None):
        if not movie:
            return None
        payload = {
            "id": movie["id"],
            "title": movie["title"],
            "overview": movie["overview"],
            "tagline": movie["tagline"],
            "original_language": movie.get("original_language", ""),
            "country_codes": movie.get("country_codes", []),
            "industry_key": movie.get("industry_key", ""),
            "industry_label": movie.get("industry_label", ""),
            "genres": movie["genres"],
            "keywords": movie["keywords"][:10],
            "cast": movie["cast"][:5],
            "directors": movie["directors"],
            "release_year": movie["release_year"],
            "runtime": movie["runtime"],
            "vote_average": round(movie["vote_average"], 1),
            "vote_count": movie["vote_count"],
            "poster_url": movie["poster_url"],
            "backdrop_url": movie["backdrop_url"],
            "trailer_key": movie["trailer_key"],
        }
        if score is not None:
            payload["score"] = score
        return payload

    def _build_query_vector(self, tokens: list[str]) -> dict:
        tf = Counter(tokens)
        max_tf = max(tf.values(), default=1)
        return {
            token: ((0.5 + 0.5 * count / max_tf) * self.idf.get(token, 1.0))
            for token, count in tf.items()
        }

    def _norm(self, vector: dict) -> float:
        return math.sqrt(sum(weight * weight for weight in vector.values()))

    def _quality_score(self, movie: dict) -> float:
        rating = min(movie["vote_average"] / 10, 1.0)
        votes = min(math.log10(movie["vote_count"] + 1) / 3.5, 1.0)
        popularity = min(math.log10(movie["popularity"] + 1) / 2.7, 1.0)
        return rating * 0.45 + votes * 0.2 + popularity * 0.35

    def _extract_country_codes(self, movie: dict) -> list[str]:
        if movie.get("country_codes"):
            return [code.upper() for code in movie.get("country_codes", []) if code]
        codes = []
        for item in movie.get("production_countries", []) or []:
            if isinstance(item, str) and item:
                codes.append(item.upper())
            elif isinstance(item, dict) and item.get("iso_3166_1"):
                codes.append(item["iso_3166_1"].upper())
        return codes

    def _industry_profile(self, original_language: str, country_codes: list[str]) -> tuple[str, str]:
        language = (original_language or "").lower()
        countries = {code.upper() for code in country_codes if code}

        if language == "hi":
            return "bollywood", "Bollywood"
        if language == "te" and "IN" in countries:
            return "tollywood", "Tollywood"
        if language == "ta" and "IN" in countries:
            return "kollywood", "Kollywood"
        if language == "ml" and "IN" in countries:
            return "mollywood", "Mollywood"
        if language == "kn" and "IN" in countries:
            return "sandalwood", "Sandalwood"
        if language == "bn" and "IN" in countries:
            return "bengali_india", "Bengali Cinema"
        if language == "en" and ("US" in countries or not countries):
            return "hollywood", "Hollywood"
        if language == "ko":
            return "korean", "Korean Cinema"
        if language == "ja":
            return "japanese", "Japanese Cinema"
        if language == "zh":
            return "chinese", "Chinese Cinema"
        if language and language in LANGUAGE_LABELS:
            return f"lang:{language}", f"{LANGUAGE_LABELS[language]} Cinema"
        if language:
            return f"lang:{language}", f"{language.upper()} Cinema"
        if countries:
            country = sorted(countries)[0]
            return f"country:{country}", f"{country} Cinema"
        return "global", "Global Cinema"

    def _same_industry(self, anchor_movie: dict, candidate_movie: dict) -> bool:
        anchor_key = anchor_movie.get("industry_key")
        candidate_key = candidate_movie.get("industry_key")
        if anchor_key and candidate_key:
            return anchor_key == candidate_key
        anchor_language = anchor_movie.get("original_language")
        candidate_language = candidate_movie.get("original_language")
        return bool(anchor_language and candidate_language and anchor_language == candidate_language)

    def _title_match_score(self, query: str, title: str) -> float:
        normalized_query = self._normalize_text(query)
        normalized_title = self._normalize_text(title)
        if not normalized_query or not normalized_title:
            return 0.0
        if normalized_query == normalized_title:
            return 1.0
        if normalized_query in normalized_title or normalized_title in normalized_query:
            return 0.92
        return SequenceMatcher(None, normalized_query, normalized_title).ratio()

    def _normalize_text(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


class AppState:
    def __init__(self):
        self.engine = RecommendationEngine()
        self.refresh_error = ""
        self.refreshing = False
        self.refresh_lock = threading.Lock()
        self._load_cached()

    def _load_cached(self):
        if CACHE_FILE.exists():
            try:
                payload = json.loads(CACHE_FILE.read_text())
                self.engine.load(payload)
            except Exception:
                self.refresh_error = "Existing cache could not be loaded."

    def should_refresh(self) -> bool:
        return not CACHE_FILE.exists() or (time.time() - self.engine.dataset["fetched_at"] > CACHE_TTL_SECONDS)

    def refresh(self, force: bool = False):
        with self.refresh_lock:
            if self.refreshing:
                return
            if not force and not self.should_refresh() and self.engine.dataset["movies"]:
                return

            api_key = load_api_key()
            if not api_key:
                self.refresh_error = "TMDB_API_KEY is missing. Add it in .env or your shell."
                return

            self.refreshing = True

        try:
            payload = TMDBClient(api_key).fetch_catalog()
            DATA_DIR.mkdir(exist_ok=True)
            CACHE_FILE.write_text(json.dumps(payload, indent=2))
            self.engine.load(payload)
            self.refresh_error = ""
        except Exception as exc:
            self.refresh_error = f"Catalog refresh failed: {exc}"
        finally:
            self.refreshing = False

    def refresh_async(self, force: bool = False):
        threading.Thread(target=self.refresh, kwargs={"force": force}, daemon=True).start()

    def discover(self, query: str, genres: list[str] | None = None):
        genres = genres or []
        if not query.strip():
            return {
                "mode": "recommendations",
                "anchor": None,
                "results": self.engine.recommend(query="", genres=genres),
            }

        api_key = load_api_key()
        collection_intent = self._match_collection_intent(query)
        if collection_intent:
            if api_key:
                try:
                    client = TMDBClient(api_key)
                    return self._discover_collection(client, collection_intent, genres=genres)
                except Exception:
                    pass
            local_collection = self._local_collection_fallback(collection_intent, genres=genres)
            if local_collection.get("results"):
                return local_collection

        local_match = self.engine.best_title_match(query)
        if local_match:
            return self.recommend_from_movie(local_match["id"], genres=genres)

        if not api_key:
            return {
                "mode": "recommendations",
                "anchor": None,
                "results": self.engine.recommend(query=query, genres=genres),
            }

        try:
            client = TMDBClient(api_key)
            live_results = client.search_movies(query)
            if live_results:
                best_match = max(
                    live_results,
                    key=lambda item: self._live_search_score(query, item),
                )
                movie_id = best_match.get("id")
                if movie_id:
                    details = client.movie_graph_bundle(movie_id)
                    anchor_movie = self.engine._process_movie(details)
                    if anchor_movie:
                        results = self._recommend_from_live_anchor(client, details, anchor_movie, genres=genres)
                        return self._movie_match_payload(anchor_movie, results)
        except Exception:
            pass

        return {
            "mode": "recommendations",
            "anchor": None,
            "results": self.engine.recommend(query=query, genres=genres),
        }

    def recommend_from_movie(self, movie_id: int, genres: list[str] | None = None):
        genres = genres or []
        api_key = load_api_key()
        local_anchor = self.engine.movie_map.get(movie_id)

        if api_key:
            try:
                client = TMDBClient(api_key)
                details = client.movie_graph_bundle(movie_id)
                anchor_movie = self.engine._process_movie(details)
                if anchor_movie:
                    results = self._recommend_from_live_anchor(client, details, anchor_movie, genres=genres)
                    if results:
                        return self._movie_match_payload(anchor_movie, results)
            except Exception:
                pass

        if local_anchor:
            local_results = self._recommend_from_external_anchor(local_anchor, genres=genres)
            return self._movie_match_payload(local_anchor, self._same_industry_first(local_anchor, local_results))

        return {
            "mode": "recommendations",
            "anchor": None,
            "results": [],
        }

    def _movie_match_payload(self, anchor_movie: dict, results: list[dict]):
        same_language_results, cross_language_results = self._split_language_groups(anchor_movie, results)
        anchor_language_code = anchor_movie.get("original_language", "")
        anchor_language_label = LANGUAGE_LABELS.get(anchor_language_code, anchor_language_code.upper() if anchor_language_code else "Original")
        ordered_results = same_language_results + cross_language_results
        return {
            "mode": "movie_match",
            "anchor": self.engine._movie_payload(anchor_movie),
            "results": ordered_results,
            "same_language_results": same_language_results,
            "cross_language_results": cross_language_results,
            "language_breakdown": {
                "anchor_language_code": anchor_language_code,
                "anchor_language_label": anchor_language_label,
                "same_language_count": len(same_language_results),
                "cross_language_count": len(cross_language_results),
            },
        }

    def _split_language_groups(self, anchor_movie: dict, results: list[dict]):
        anchor_language = anchor_movie.get("original_language", "")
        same_language_results = []
        cross_language_results = []
        seen_ids = set()

        for item in results:
            movie_id = item.get("id")
            if not movie_id or movie_id in seen_ids or movie_id == anchor_movie.get("id"):
                continue
            seen_ids.add(movie_id)
            if anchor_language and item.get("original_language") == anchor_language:
                same_language_results.append(item)
            else:
                cross_language_results.append(item)

        return same_language_results, cross_language_results

    def _match_collection_intent(self, query: str):
        normalized_query = self.engine._normalize_text(query)
        collection_intents = [
            (
                [
                    "animated movie",
                    "animated movies",
                    "animation movie",
                    "animation movies",
                    "animated film",
                    "animated films",
                    "cartoon movie",
                    "cartoon movies",
                ],
                {
                    "label": "Animation Picks",
                    "description": "Animation-only movies discovered live from TMDB.",
                    "params": {
                        "with_genres": "16",
                        "sort_by": "popularity.desc",
                        "page": 1,
                    },
                    "strict_genres": ["Animation"],
                },
            ),
            (
                ["bollywood", "hindi movie", "hindi movies", "hindi cinema"],
                {
                    "label": "Bollywood Picks",
                    "description": "Popular Hindi-language movies discovered live from TMDB.",
                    "params": {
                        "with_original_language": "hi",
                        "sort_by": "popularity.desc",
                        "page": 1,
                    },
                },
            ),
            (
                ["hollywood", "english movie", "english movies", "hollywood movie", "hollywood movies"],
                {
                    "label": "Hollywood Picks",
                    "description": "Popular English-language studio movies discovered live from TMDB.",
                    "params": {
                        "with_original_language": "en",
                        "sort_by": "popularity.desc",
                        "page": 1,
                    },
                },
            ),
            (
                ["korean movie", "korean movies", "korean cinema", "k cinema"],
                {
                    "label": "Korean Picks",
                    "description": "Popular Korean-language movies discovered live from TMDB.",
                    "params": {
                        "with_original_language": "ko",
                        "sort_by": "popularity.desc",
                        "page": 1,
                    },
                },
            ),
            (
                ["anime movie", "anime movies", "anime film", "anime films"],
                {
                    "label": "Anime Picks",
                    "description": "Japanese animation movies discovered live from TMDB.",
                    "params": {
                        "with_original_language": "ja",
                        "with_genres": "16",
                        "sort_by": "popularity.desc",
                        "page": 1,
                    },
                    "strict_genres": ["Animation"],
                },
            ),
            (
                ["japanese movie", "japanese movies", "japanese cinema"],
                {
                    "label": "Japanese Picks",
                    "description": "Popular Japanese-language movies discovered live from TMDB.",
                    "params": {
                        "with_original_language": "ja",
                        "sort_by": "popularity.desc",
                        "page": 1,
                    },
                },
            ),
        ]
        for phrases, intent in collection_intents:
            if any(phrase in normalized_query for phrase in phrases):
                return intent
        return None

    def _discover_collection(self, client: TMDBClient, intent: dict, genres: list[str] | None = None):
        genres = genres or []
        strict_genres = intent.get("strict_genres", [])
        live_results = client.discover_movies(**intent["params"])[:18]
        movie_ids = [movie.get("id") for movie in live_results if movie.get("id")]
        detailed_movies = self._load_live_movie_details(client, movie_ids[:12])
        if strict_genres:
            detailed_movies = self._filter_movies_by_required_genres(detailed_movies, strict_genres)

        if detailed_movies:
            ranked = self._rank_collection_details(detailed_movies, genres=genres)
            return {
                "mode": "collection_match",
                "anchor": ranked[0] if ranked else None,
                "collection_label": intent["label"],
                "collection_description": intent["description"],
                "results": ranked,
            }

        cards = [self.engine._normalize_card(movie) for movie in live_results]
        cards = [card for card in cards if card]
        if strict_genres:
            cards = self._filter_movies_by_required_genres(cards, strict_genres)
        return {
            "mode": "collection_match",
            "anchor": cards[0] if cards else None,
            "collection_label": intent["label"],
            "collection_description": intent["description"],
            "results": cards,
        }

    def _local_collection_fallback(self, intent: dict, genres: list[str] | None = None):
        genres = genres or []
        strict_genres = intent.get("strict_genres", [])
        language_code = intent.get("params", {}).get("with_original_language", "")

        with self.engine.lock:
            movies = list(self.engine.dataset["movies"])

        filtered_movies = []
        for movie in movies:
            if language_code and movie.get("original_language") != language_code:
                continue
            if strict_genres and not self._movie_has_required_genres(movie, strict_genres):
                continue
            filtered_movies.append(movie)

        ranked = self._rank_collection_details(filtered_movies, genres=genres)
        return {
            "mode": "collection_match",
            "anchor": ranked[0] if ranked else None,
            "collection_label": intent["label"],
            "collection_description": f"{intent['description']} Using the local catalog fallback.",
            "results": ranked,
        }

    def _load_live_movie_details(self, client: TMDBClient, movie_ids: list[int]):
        details = []
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {
                executor.submit(client.movie_details, movie_id): movie_id
                for movie_id in movie_ids
            }
            for future in as_completed(futures):
                try:
                    payload = future.result()
                    processed = self.engine._process_movie(payload)
                    if processed:
                        details.append(processed)
                except Exception:
                    continue
        return details

    def _movie_has_required_genres(self, movie: dict, required_genres: list[str]):
        movie_genres = set(movie.get("genres", []))
        return all(required_genre in movie_genres for required_genre in required_genres)

    def _filter_movies_by_required_genres(self, movies: list[dict], required_genres: list[str]):
        return [movie for movie in movies if self._movie_has_required_genres(movie, required_genres)]

    def _rank_collection_details(self, detailed_movies: list[dict], genres: list[str] | None = None):
        genres = genres or []
        ranked = []
        for movie in detailed_movies:
            score = self.engine._quality_score(movie)
            if genres:
                score += len(set(genres) & set(movie["genres"])) * 0.08
            ranked.append((score, movie))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [self.engine._movie_payload(movie, score=round(score, 4)) for score, movie in ranked]

    def _recommend_from_external_anchor(self, anchor_movie: dict, genres: list[str] | None = None):
        genres = genres or []
        tf = Counter(anchor_movie["tokens"])
        max_tf = max(tf.values(), default=1)
        anchor_vector = {
            token: ((0.5 + 0.5 * count / max_tf) * self.engine.idf.get(token, 1.0))
            for token, count in tf.items()
        }
        anchor_norm = math.sqrt(sum(weight * weight for weight in anchor_vector.values()))
        scores = []
        with self.engine.lock:
            movies = list(self.engine.dataset["movies"])
        for movie in movies:
            score = cosine_similarity(anchor_vector, anchor_norm, movie["vector"], movie["norm"]) * 0.58
            if movie["id"] in anchor_movie["recommendation_ids"]:
                score += 0.12
            if movie["id"] in anchor_movie["similar_ids"]:
                score += 0.08
            score += 0.45 if self.engine._same_industry(anchor_movie, movie) else -0.18
            overlap = len(set(anchor_movie["genres"]) & set(movie["genres"]))
            score += min(overlap * 0.035, 0.14)
            if genres:
                score += len(set(genres) & set(movie["genres"])) * 0.04
            score += self.engine._quality_score(movie) * 0.22
            if score > 0:
                scores.append((score, movie))
        scores.sort(key=lambda item: item[0], reverse=True)
        return [self.engine._movie_payload(movie, score=round(score, 4)) for score, movie in scores[:24]]

    def _same_industry_first(self, anchor_movie: dict, primary_items: list[dict], fallback_items: list[dict] | None = None):
        fallback_items = fallback_items or []
        seen_ids = set()
        same_industry = []
        cross_industry = []

        def bucket(items: list[dict]):
            for item in items:
                movie_id = item.get("id")
                if not movie_id or movie_id in seen_ids or movie_id == anchor_movie.get("id"):
                    continue
                seen_ids.add(movie_id)
                if self.engine._same_industry(anchor_movie, item):
                    same_industry.append(item)
                else:
                    cross_industry.append(item)

        bucket(primary_items)
        bucket(fallback_items)
        return (same_industry + cross_industry)[:24]

    def _recommend_from_live_anchor(
        self,
        client: TMDBClient,
        raw_anchor: dict,
        anchor_movie: dict,
        genres: list[str] | None = None,
    ):
        genres = genres or []
        candidate_entries = {}

        for rank, item in enumerate(raw_anchor.get("recommendations", {}).get("results", [])[:14]):
            movie_id = item.get("id")
            if not movie_id or movie_id == anchor_movie["id"]:
                continue
            entry = candidate_entries.setdefault(movie_id, {"raw": item, "score": 0.0})
            entry["score"] += max(0.42 - rank * 0.015, 0.2)

        for rank, item in enumerate(raw_anchor.get("similar", {}).get("results", [])[:14]):
            movie_id = item.get("id")
            if not movie_id or movie_id == anchor_movie["id"]:
                continue
            entry = candidate_entries.setdefault(movie_id, {"raw": item, "score": 0.0})
            entry["score"] += max(0.08 - rank * 0.004, 0.02)

        movie_ids = list(candidate_entries.keys())[:18]
        detailed_movies = self._load_live_movie_details(client, movie_ids)
        detail_map = {movie["id"]: movie for movie in detailed_movies}

        anchor_vector = self._build_live_vector(anchor_movie)
        anchor_norm = self.engine._norm(anchor_vector)
        ranked = []

        for movie_id, entry in candidate_entries.items():
            score = entry["score"]
            movie = detail_map.get(movie_id)
            if movie:
                candidate_vector = self._build_live_vector(movie)
                candidate_norm = self.engine._norm(candidate_vector)
                semantic_score = cosine_similarity(anchor_vector, anchor_norm, candidate_vector, candidate_norm)
                genre_overlap = min(len(set(anchor_movie["genres"]) & set(movie["genres"])) * 0.05, 0.2)
                keyword_overlap = min(len(set(anchor_movie["keywords"]) & set(movie["keywords"])) * 0.03, 0.12)
                director_overlap = 0.08 if set(anchor_movie["directors"]) & set(movie["directors"]) else 0.0
                cast_overlap = min(len(set(anchor_movie["cast"]) & set(movie["cast"])) * 0.02, 0.08)
                language_bonus = 0.1 if anchor_movie.get("original_language") and anchor_movie.get("original_language") == movie.get("original_language") else 0.0
                industry_bonus = 0.6 if self.engine._same_industry(anchor_movie, movie) else -0.25
                user_genre_bonus = len(set(genres) & set(movie["genres"])) * 0.03 if genres else 0.0
                quality_bonus = self.engine._quality_score(movie) * 0.08

                score += (
                    semantic_score * 0.28
                    + genre_overlap
                    + keyword_overlap
                    + director_overlap
                    + cast_overlap
                    + language_bonus
                    + industry_bonus
                    + user_genre_bonus
                    + quality_bonus
                )
                ranked.append((score, self.engine._movie_payload(movie, score=round(score, 4))))
            else:
                card = self.engine._normalize_card(entry["raw"])
                if card:
                    card["score"] = round(score, 4)
                    ranked.append((score, card))

        ranked.sort(key=lambda item: item[0], reverse=True)
        payloads = [payload for _, payload in ranked[:24]]
        local_fallbacks = self._recommend_from_external_anchor(anchor_movie, genres=genres)
        return self._same_industry_first(anchor_movie, payloads, fallback_items=local_fallbacks)

    def _live_search_score(self, query: str, item: dict) -> float:
        title = item.get("title", "")
        original_title = item.get("original_title", "")
        release_date = item.get("release_date") or ""
        recency_penalty = 0.02 if release_date >= "2020-01-01" else 0.0
        title_score = self.engine._title_match_score(query, title)
        original_score = self.engine._title_match_score(query, original_title)
        popularity = min(math.log10((item.get("popularity") or 0) + 1) / 3, 1.0)
        votes = min(math.log10((item.get("vote_count") or 0) + 1) / 4, 1.0)
        return max(title_score, original_score) * 0.75 + popularity * 0.15 + votes * 0.1 - recency_penalty

    def _build_live_vector(self, movie: dict):
        tf = Counter(movie["tokens"])
        max_tf = max(tf.values(), default=1)
        return {
            token: ((0.5 + 0.5 * count / max_tf) * self.engine.idf.get(token, 1.0))
            for token, count in tf.items()
        }


APP_STATE = AppState()


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class MovieRequestHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path == "/":
            return self._send_head_response(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        if path.startswith("/static/"):
            relative_path = path.replace("/static/", "", 1)
            file_path = (STATIC_DIR / relative_path).resolve()
            if not str(file_path).startswith(str(STATIC_DIR.resolve())) or not file_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND, "Asset not found")
                return
            content_type = "text/plain; charset=utf-8"
            if file_path.suffix == ".css":
                content_type = "text/css; charset=utf-8"
            elif file_path.suffix == ".js":
                content_type = "application/javascript; charset=utf-8"
            return self._send_head_response(file_path, content_type)
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/":
            return self._serve_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        if path.startswith("/static/"):
            relative_path = path.replace("/static/", "", 1)
            return self._serve_static_asset(relative_path)
        if path == "/api/bootstrap":
            if APP_STATE.should_refresh() and not APP_STATE.refreshing:
                APP_STATE.refresh_async(force=True)
            payload = APP_STATE.engine.bootstrap()
            payload["status"] = {
                "refreshing": APP_STATE.refreshing,
                "error": APP_STATE.refresh_error,
            }
            return self._send_json(payload)
        if path == "/api/search":
            search_query = (query.get("q") or [""])[0]
            return self._send_json({"results": APP_STATE.engine.search(search_query)})
        if path == "/api/discover":
            free_text = (query.get("q") or [""])[0]
            genres = [item for item in (query.get("genres") or [""])[0].split(",") if item]
            return self._send_json(APP_STATE.discover(free_text, genres=genres))
        if path == "/api/recommend":
            movie_id = (query.get("movie_id") or [None])[0]
            free_text = (query.get("q") or [""])[0]
            genres = [item for item in (query.get("genres") or [""])[0].split(",") if item]
            movie_id_value = int(movie_id) if movie_id and str(movie_id).isdigit() else None
            if movie_id_value is not None:
                return self._send_json(APP_STATE.recommend_from_movie(movie_id_value, genres=genres))
            return self._send_json(
                {
                    "results": APP_STATE.engine.recommend(
                        movie_id=movie_id_value,
                        query=free_text,
                        genres=genres,
                    )
                }
            )
        if path == "/api/refresh":
            APP_STATE.refresh(force=True)
            return self._send_json({"ok": True, "error": APP_STATE.refresh_error})

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, format, *args):
        return

    def _serve_static_asset(self, relative_path: str):
        file_path = (STATIC_DIR / relative_path).resolve()
        if not str(file_path).startswith(str(STATIC_DIR.resolve())) or not file_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "Asset not found")
            return
        content_type = "text/plain; charset=utf-8"
        if file_path.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        elif file_path.suffix == ".js":
            content_type = "application/javascript; charset=utf-8"
        elif file_path.suffix == ".json":
            content_type = "application/json; charset=utf-8"
        self._serve_file(file_path, content_type)

    def _serve_file(self, file_path: Path, content_type: str):
        if not file_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        content = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_head_response(self, file_path: Path, content_type: str):
        if not file_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        content_length = file_path.stat().st_size
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        self.end_headers()

    def _send_json(self, payload: dict):
        content = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def ensure_seed_files():
    DATA_DIR.mkdir(exist_ok=True)
    if not CONFIG_FILE.exists():
        CONFIG_FILE.write_text("TMDB_API_KEY=\n")


def is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def find_available_port(host: str, preferred_port: int, attempts: int = 20) -> int | None:
    if not is_port_open(host, preferred_port):
        return preferred_port
    for port in range(preferred_port + 1, preferred_port + attempts + 1):
        if not is_port_open(host, port):
            return port
    return None


def is_app_responding(host: str, port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/", timeout=2) as response:
            content_type = response.headers.get("Content-Type", "")
            return response.status == 200 and "text/html" in content_type
    except Exception:
        return False


def main():
    ensure_seed_files()
    if APP_STATE.should_refresh():
        APP_STATE.refresh_async(force=True)
    selected_port = find_available_port(HOST, PORT)
    if selected_port is None:
        raise RuntimeError(f"No free port found starting from {PORT}.")

    if selected_port != PORT and is_app_responding(HOST, PORT):
        print(f"Moviesss AI is already running on http://{HOST}:{PORT}")
        print(f"Use PORT={selected_port} python3 app.py if you want a second instance.")
        return

    if selected_port != PORT:
        print(f"Port {PORT} is busy. Starting Moviesss AI on http://{HOST}:{selected_port} instead.")

    server = ReusableThreadingHTTPServer((HOST, selected_port), MovieRequestHandler)
    print(f"Server running on http://{HOST}:{selected_port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
