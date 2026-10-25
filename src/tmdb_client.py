"""Small client for fetching movie metadata from the TMDB API."""

import os
from pathlib import Path

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

TMDB_BASE_URL = "https://api.themoviedb.org/3"


def get_movie_details(tmdb_id: int) -> dict:
    """Fetch one movie's metadata from TMDB using its TMDB identifier."""

    if not isinstance(tmdb_id, int) or isinstance(tmdb_id, bool) or tmdb_id <= 0:
        raise ValueError("tmdb_id must be a positive integer.")

    token = os.getenv("TMDB_READ_ACCESS_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "TMDB_READ_ACCESS_TOKEN is missing. Add it to the project's .env file."
        )

    headers = {
        "Authorization": f"Bearer {token}",
        "accept": "application/json",
    }
    url = f"{TMDB_BASE_URL}/movie/{tmdb_id}"

    response = requests.get(
        url,
        headers=headers,
        timeout=15
    )
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    movie = get_movie_details(862)
    fields = ["title", "release_date", "genres", "overview", "vote_average"]
    selected_movie = {
        key: movie.get(key)
        for key in fields
    }
    print(selected_movie)
