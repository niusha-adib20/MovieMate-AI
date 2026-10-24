"""Tests for the popularity-based, genre recommendation utilities."""

import pytest

from src.content_based import load_processed_movies
from src.recommender import (
    load_movie_rating_stats,
    merge_movies_with_ratings,
    calculate_popularity_score,
    recommend_by_genre
)

@pytest.fixture(scope="module")
def genre_data():
    """Load and score the catalogue once for all tests in this module."""

    movies = load_processed_movies()
    rating_stats = load_movie_rating_stats()
    movies_with_ratings = merge_movies_with_ratings(movies, rating_stats)
    scored_movies = calculate_popularity_score(movies_with_ratings)
    return scored_movies


def test_recommend_by_genre_action(genre_data):
    """Action recommendations should contain the requested result fields."""

    result = recommend_by_genre("Action", genre_data, top_n=5)

    assert len(result) == 5
    assert "clean_title" in result.columns
    assert "average_rating" in result.columns
    assert "popularity_score" in result.columns


def test_recommend_by_genre_ignores_case_and_surrounding_spaces(genre_data):
    """Genre input should be normalized before recommendations are generated."""

    expected_result = recommend_by_genre("Action", genre_data, top_n=5)
    spaced_result = recommend_by_genre(" action ", genre_data, top_n=5)
    uppercase_result = recommend_by_genre("ACTION", genre_data, top_n=5)

    assert expected_result.equals(spaced_result)
    assert expected_result.equals(uppercase_result)


def test_recommend_by_genre_rejects_invalid_or_empty_genres(genre_data):
    """Invalid user input should produce clear errors instead of empty results."""

    with pytest.raises(ValueError):
        recommend_by_genre("NotARealGenre", genre_data, top_n=5)

    with pytest.raises(ValueError):
        recommend_by_genre("", genre_data, top_n=5)


def test_recommend_by_genre_sorts_by_popularity_score(genre_data):
    """The highest-scoring movies should appear first in the result."""

    result = recommend_by_genre("Action", genre_data, top_n=5)

    assert result["popularity_score"].is_monotonic_decreasing
