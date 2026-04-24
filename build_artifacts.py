import ast
import pickle
import re
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
NLTK_DATA_DIR = BASE_DIR / "nltk_data"

MOVIES_PATH = RAW_DATA_DIR / "movies_metadata.csv"
CREDITS_PATH = RAW_DATA_DIR / "credits.csv"
KEYWORDS_PATH = RAW_DATA_DIR / "keywords.csv"

DIRECTOR_WEIGHT = 6
GENRE_WEIGHT = 3
CAST_WEIGHT = 2
KEYWORD_WEIGHT = 3
MAX_CAST = 5
MAX_GENRES = 4
MAX_KEYWORDS = 8


def ensure_nltk_resource(resource: str, *locators: str) -> None:
    for locator in locators:
        try:
            nltk.data.find(locator)
            return
        except LookupError:
            continue

    NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    nltk.download(resource, download_dir=str(NLTK_DATA_DIR), quiet=True)


def configure_nltk() -> tuple[set[str], WordNetLemmatizer]:
    if str(NLTK_DATA_DIR) not in nltk.data.path:
        nltk.data.path.insert(0, str(NLTK_DATA_DIR))

    ensure_nltk_resource("stopwords", "corpora/stopwords", "corpora/stopwords.zip")
    ensure_nltk_resource("wordnet", "corpora/wordnet", "corpora/wordnet.zip")
    ensure_nltk_resource("omw-1.4", "corpora/omw-1.4", "corpora/omw-1.4.zip")

    return set(stopwords.words("english")), WordNetLemmatizer()


def parse_literal_list(raw_value: str) -> list[dict]:
    if pd.isna(raw_value):
        return []

    try:
        value = ast.literal_eval(raw_value)
    except (SyntaxError, ValueError):
        return []

    return value if isinstance(value, list) else []


def compact_name(name: str) -> str:
    return re.sub(r"\s+", "", str(name).strip())


def extract_names(raw_value: str, limit: int) -> list[str]:
    names: list[str] = []
    for item in parse_literal_list(raw_value):
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not name:
            continue
        names.append(compact_name(name))
        if len(names) >= limit:
            break
    return names


def extract_director(raw_value: str) -> list[str]:
    for item in parse_literal_list(raw_value):
        if not isinstance(item, dict):
            continue
        if item.get("job") == "Director" and item.get("name"):
            return [compact_name(item["name"])]
    return []


def preprocess_text(
    text: str, stop_words: set[str], lemmatizer: WordNetLemmatizer
) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    tokens: list[str] = []
    for token in text.split():
        if token in stop_words or len(token) < 2:
            continue
        tokens.append(lemmatizer.lemmatize(token))

    return " ".join(tokens)


def read_source_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [path.name for path in (MOVIES_PATH, CREDITS_PATH, KEYWORDS_PATH) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing raw dataset files in data/raw: " + ", ".join(missing)
        )

    movies = pd.read_csv(MOVIES_PATH, low_memory=False)
    credits = pd.read_csv(CREDITS_PATH)
    keywords = pd.read_csv(KEYWORDS_PATH)

    return movies, credits, keywords


def clean_ids(frame: pd.DataFrame, column: str = "id") -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = cleaned.dropna(subset=[column]).copy()
    cleaned[column] = cleaned[column].astype("int64")
    return cleaned


def build_feature_frame(
    movies: pd.DataFrame,
    credits: pd.DataFrame,
    keywords: pd.DataFrame,
    stop_words: set[str],
    lemmatizer: WordNetLemmatizer,
) -> pd.DataFrame:
    movies = clean_ids(movies)
    credits = clean_ids(credits)
    keywords = clean_ids(keywords)

    frame = (
        movies[
            [
                "id",
                "title",
                "overview",
                "genres",
                "vote_average",
                "popularity",
            ]
        ]
        .merge(credits[["id", "cast", "crew"]], on="id", how="left")
        .merge(keywords[["id", "keywords"]], on="id", how="left")
    )

    frame = frame.dropna(subset=["title"]).drop_duplicates(subset=["title"]).reset_index(
        drop=True
    )
    frame["overview"] = frame["overview"].fillna("")

    frame["genre_terms"] = frame["genres"].apply(
        lambda value: extract_names(value, limit=MAX_GENRES)
    )
    frame["cast_terms"] = frame["cast"].apply(
        lambda value: extract_names(value, limit=MAX_CAST)
    )
    frame["director_terms"] = frame["crew"].apply(extract_director)
    frame["keyword_terms"] = frame["keywords"].apply(
        lambda value: extract_names(value, limit=MAX_KEYWORDS)
    )

    frame["genres"] = frame["genre_terms"].apply(lambda values: " ".join(values))
    frame["cast"] = frame["cast_terms"].apply(lambda values: " ".join(values))
    frame["director"] = frame["director_terms"].apply(lambda values: " ".join(values))
    frame["keywords"] = frame["keyword_terms"].apply(lambda values: " ".join(values))

    frame["raw_tags"] = frame.apply(
        lambda row: " ".join(
            [
                row["overview"],
                " ".join(row["genre_terms"] * GENRE_WEIGHT),
                " ".join(row["cast_terms"] * CAST_WEIGHT),
                " ".join(row["director_terms"] * DIRECTOR_WEIGHT),
                " ".join(row["keyword_terms"] * KEYWORD_WEIGHT),
            ]
        ).strip(),
        axis=1,
    )

    frame["tags"] = frame["raw_tags"].apply(
        lambda text: preprocess_text(text, stop_words=stop_words, lemmatizer=lemmatizer)
    )

    return frame[
        [
            "id",
            "title",
            "overview",
            "genres",
            "cast",
            "director",
            "keywords",
            "vote_average",
            "popularity",
            "tags",
        ]
    ]


def build_and_save_artifacts(output_dir: Path = BASE_DIR) -> pd.DataFrame:
    stop_words, lemmatizer = configure_nltk()
    movies, credits, keywords = read_source_data()

    frame = build_feature_frame(
        movies=movies,
        credits=credits,
        keywords=keywords,
        stop_words=stop_words,
        lemmatizer=lemmatizer,
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(frame["tags"])
    indices = pd.Series(frame.index, index=frame["title"]).drop_duplicates()

    artifacts = {
        "df.pkl": frame,
        "indices.pkl": indices,
        "tfidf.pkl": vectorizer,
        "tfidf_matrix.pkl": tfidf_matrix,
    }

    for filename, artifact in artifacts.items():
        with (output_dir / filename).open("wb") as handle:
            pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return frame


if __name__ == "__main__":
    dataset = build_and_save_artifacts()
    print(
        f"Saved recommendation artifacts for {len(dataset):,} unique movie titles to {BASE_DIR}"
    )
