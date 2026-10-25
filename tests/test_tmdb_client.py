"""Unit tests for the TMDB API client without making network requests."""

from unittest.mock import Mock

import pytest

import src.tmdb_client as tmdb_client


def test_get_movie_details_returns_api_data(monkeypatch):
    """The client should return the JSON body from a successful response."""

    monkeypatch.setenv("TMDB_READ_ACCESS_TOKEN", "fake-token")

    expected_data = {
        "title": "Toy Story",
        "vote_average": 7.99,
    }

    mock_response = Mock()
    mock_response.json.return_value = expected_data
    mock_get = Mock(return_value=mock_response)
    monkeypatch.setattr(tmdb_client.requests, "get", mock_get)

    result = tmdb_client.get_movie_details(862)

    assert result == expected_data
    mock_response.raise_for_status.assert_called_once_with()
    mock_get.assert_called_once_with(
        "https://api.themoviedb.org/3/movie/862",
        headers={
            "Authorization": "Bearer fake-token",
            "accept": "application/json",
        },
        timeout=15,
    )


@pytest.mark.parametrize("invalid_tmdb_id", [0, -1, "862", True])
def test_get_movie_details_rejects_invalid_ids(invalid_tmdb_id):
    """Only positive integer TMDB identifiers are valid."""

    with pytest.raises(ValueError):
        tmdb_client.get_movie_details(invalid_tmdb_id)


def test_get_movie_details_requires_token(monkeypatch):
    """The client should stop before making a request when the token is absent."""

    monkeypatch.delenv("TMDB_READ_ACCESS_TOKEN", raising=False)

    with pytest.raises(RuntimeError):
        tmdb_client.get_movie_details(862)
