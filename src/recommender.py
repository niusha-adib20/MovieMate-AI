"""Recommendation utilities for MovieMate AI."""

import pandas as pd

from src.data_processing import PROCESSED_RATING_STATS_PATH


def load_movie_rating_stats() -> pd.DataFrame:
    """Load per-movie rating counts and average ratings from processed data."""

    if not PROCESSED_RATING_STATS_PATH.exists():
        raise FileNotFoundError(
            "Movie rating statistics file not found: "
            f"{PROCESSED_RATING_STATS_PATH}"
        )

    return pd.read_csv(PROCESSED_RATING_STATS_PATH)


def merge_movies_with_ratings(
    movies: pd.DataFrame,
    rating_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Join movie metadata with per-movie rating statistics.

    Movies without ratings are excluded because they cannot be ranked by
    rating-based popularity.
    """

    movies_with_ratings = movies.merge(
        rating_stats,
        on="movieId",
        how="inner",
        validate="one_to_one",
    )

    return movies_with_ratings


def calculate_popularity_score(
    movies_with_ratings: pd.DataFrame,
) -> pd.DataFrame:
    """Add an IMDb-style weighted popularity score to rated movies.

    A movie's average rating is smoothed toward the global weighted mean when
    it has relatively few ratings, preventing single-vote movies from ranking
    unrealistically high. The 90th percentile of rating counts is used as the
    smoothing threshold because MovieLens ratings have a long-tailed spread.
    """

    scored_movies = movies_with_ratings.copy()
    global_mean_rating = (
        (
            scored_movies["average_rating"]
            * scored_movies["rating_count"]
        ).sum()
        / scored_movies["rating_count"].sum()
    )
    minimum_rating_count = scored_movies["rating_count"].quantile(0.90)

    vote_weight = (
        scored_movies["rating_count"]
        / (scored_movies["rating_count"] + minimum_rating_count)
    )
    scored_movies["popularity_score"] = (
        vote_weight * scored_movies["average_rating"]
        + (1 - vote_weight) * global_mean_rating
    )

    return scored_movies


def recommend_by_genre(
    genre: str,
    scored_movies: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return the highest-ranked movies that match an exact genre label.

    Genre matching ignores capitalization and surrounding whitespace. Movies
    without genre data are excluded from genre-based recommendations.
    """

    normalized_genre = genre.strip().casefold()
    if not normalized_genre:
        raise ValueError("Genre must not be empty.")

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    genre_matches = (
        scored_movies["genres"]
        .fillna("")
        .str.casefold()
        .str.split("|")
        .map(lambda genres: normalized_genre in genres)
    )
    movies_with_genre = scored_movies.loc[genre_matches]

    if movies_with_genre.empty:
        raise ValueError(f"Genre not found in the catalogue: {genre!r}")

    result_columns = [
        "clean_title",
        "release_year",
        "genres",
        "rating_count",
        "average_rating",
        "popularity_score",
    ]
    return (
        movies_with_genre.sort_values("popularity_score", ascending=False)
        .loc[:, result_columns]
        .head(top_n)
        .reset_index(drop=True)
    )
