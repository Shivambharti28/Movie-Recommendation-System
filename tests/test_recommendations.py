import pickle
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]


def load_pickle(filename: str):
    with (BASE_DIR / filename).open("rb") as handle:
        return pickle.load(handle)


def recommend(title: str, limit: int = 10) -> list[str]:
    df = load_pickle("df.pkl")
    indices = load_pickle("indices.pkl")
    matrix = load_pickle("tfidf_matrix.pkl")

    idx = int(indices[title])
    scores = (matrix @ matrix[idx].T).toarray().ravel()
    order = scores.argsort()[::-1]

    results: list[str] = []
    for candidate_idx in order:
        if int(candidate_idx) == idx:
            continue
        results.append(str(df.iloc[int(candidate_idx)]["title"]))
        if len(results) >= limit:
            break

    return results


def test_artifacts_include_richer_feature_columns():
    df = load_pickle("df.pkl")
    assert {"genres", "cast", "director", "keywords", "tags"} <= set(df.columns)


def test_inception_returns_nolan_neighbors_in_top_ten():
    recommendations = recommend("Inception", limit=10)

    assert "The Dark Knight" in recommendations
    assert "Interstellar" in recommendations
