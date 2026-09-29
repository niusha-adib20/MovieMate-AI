"""Content-based movie recommendation utilities."""

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data_processing import PROCESSED_MOVIES_PATH


def load_processed_movies() -> pd.DataFrame:
    """Load the processed movie catalogue for recommendation."""

    if not PROCESSED_MOVIES_PATH.exists():
        raise FileNotFoundError(
            f"Processed movie file not found: {PROCESSED_MOVIES_PATH}"
        )

    return pd.read_csv(PROCESSED_MOVIES_PATH)


def build_tfidf_matrix(
    movies: pd.DataFrame,
) -> tuple[TfidfVectorizer, csr_matrix]:
    """Convert movie genre text into TF-IDF feature vectors."""

    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b[\w-]+\b",
    )
    tfidf_matrix = vectorizer.fit_transform(
        movies["genres_text"].fillna("")
    )
    return vectorizer, tfidf_matrix


def recommend_similar_movies(
    title: str,
    movies: pd.DataFrame,
    tfidf_matrix: csr_matrix,
    top_n: int = 5,
) -> pd.DataFrame:
    """Return the most genre-similar movies to a title in the catalogue.

    The lookup ignores capitalization and surrounding whitespace. A movie with
    no genre data cannot receive meaningful genre-based recommendations.
    """

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    normalized_title = title.strip().casefold()
    normalized_catalogue_titles = (
        movies["clean_title"].fillna("").str.strip().str.casefold()
    )
    title_matches = normalized_catalogue_titles.eq(normalized_title)

    if not title_matches.any():
        raise ValueError(f"Movie not found in the catalogue: {title!r}")

    # The TF-IDF matrix row order matches the DataFrame row order.
    movie_position = title_matches.to_numpy().nonzero()[0][0]
    movie_vector = tfidf_matrix[movie_position]

    if movie_vector.nnz == 0:
        raise ValueError(
            f"Cannot recommend from {title!r} because it has no genre data."
        )

    similarity_scores = cosine_similarity(movie_vector, tfidf_matrix).ravel()
    ordered_indices = similarity_scores.argsort()[::-1]
    recommended_indices = ordered_indices[ordered_indices != movie_position][:top_n]

    recommendations = movies.iloc[recommended_indices][
        ["clean_title", "release_year", "genres"]
    ].copy()
    recommendations["similarity_score"] = similarity_scores[recommended_indices]

    return recommendations.reset_index(drop=True)


if __name__ == "__main__":
    movies = load_processed_movies()

    print(f"Loaded movies: {movies.shape}")
    print(movies[["clean_title", "genres_text"]].head())
    
    vectorizer, tfidf_matrix = build_tfidf_matrix(movies)

    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print("\nGenre vocabulary:")
    print(vectorizer.get_feature_names_out())
    print(recommend_similar_movies("Toy story" , movies,tfidf_matrix,5))
