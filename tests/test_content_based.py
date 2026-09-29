import pytest

from src.content_based import (
    build_tfidf_matrix,
    load_processed_movies,
    recommend_similar_movies,
)


@pytest.fixture(scope="module")
def recommendation_data():
    """Load the catalogue and build TF-IDF features once for this module."""

    movies = load_processed_movies()
    _vectorizer, tfidf_matrix = build_tfidf_matrix(movies)
    return movies, tfidf_matrix


def test_toy_story_returns_five_similar_movies(recommendation_data):
    movies, tfidf_matrix = recommendation_data
    result = recommend_similar_movies("Toy Story", movies, tfidf_matrix, top_n=5)

    assert len(result) == 5
    assert not result["clean_title"].str.casefold().eq("toy story").any()
    assert "similarity_score" in result.columns


def test_movie_not_found(recommendation_data):
    movies, tfidf_matrix = recommendation_data
    with pytest.raises(ValueError):
        recommend_similar_movies("Not A Movie", movies, tfidf_matrix)


def test_case_insensitive_title_lookup(recommendation_data):
    movies, tfidf_matrix = recommendation_data
    result_1 = (
        recommend_similar_movies("toy story", movies, tfidf_matrix, top_n=5)
    )
    result_2 = (
        recommend_similar_movies("TOY STORY", movies, tfidf_matrix, top_n=5)
    )

    assert result_1.equals(result_2)
