"""Utilities for preprocessing MovieLens data."""

from pathlib import Path
import pandas as pd 
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT/ "data" / "raw" / "ml-25m"
PROCESSED_DATA_DIR = PROJECT_ROOT/ "data" / "processed"

MOVIES_PATH = RAW_DATA_DIR/ "movies.csv"
LINKS_PATH = RAW_DATA_DIR/ "links.csv"

PROCESSED_MOVIES_PATH = PROCESSED_DATA_DIR / "movies_processed.csv"

def load_raw_data() -> tuple[pd.DataFrame , pd.DataFrame] :
    """Load the MovieLens movie catalogue and external ID mapping."""
    
    if not MOVIES_PATH.exists() :
        raise FileNotFoundError(f"Movies file not found: {MOVIES_PATH}")
    
    if not LINKS_PATH.exists() :
        raise FileNotFoundError(f"Links file not found: {LINKS_PATH}")
    
    movies = pd.read_csv(MOVIES_PATH)
    links = pd.read_csv(LINKS_PATH)
    
    return movies , links

def inspect_movies(movies: pd.DataFrame) -> None :
    """Print a compact data-quality summary for the movie catalogue."""
    
    print("\nFirst five movies:")
    print(movies.head())

    print("\nMissing values:")
    print(movies.isna().sum())

    print("\nDuplicate movie IDs:")
    print(movies["movieId"].duplicated().sum())
    
def inspect_genres(movies: pd.DataFrame) -> None :
    """Find movies whose genre is unknown but stored as text."""
    
    movies_without_genres = movies[
        movies["genres"] == "(no genres listed)"
    ]
    
    print("\nMovies without genre information:")
    print(len(movies_without_genres))

    print("\nExamples:")
    print(movies_without_genres[["movieId", "title", "genres"]].head())
    
def merge_tmdb_ids(movies: pd.DataFrame, links : pd.DataFrame) -> pd.DataFrame :
    """Attach the TMDB identifier to each MovieLens movie."""
    
    movie_catalogue = movies.merge(
        links[["movieId" , "tmdbId"]],
        on = "movieId" ,
        how="left",
        validate="one_to_one"
    )
    
    return movie_catalogue

def clean_genres(movie_catalogue: pd.DataFrame) -> pd.DataFrame :
    """Convert placeholder genre text into a proper missing value."""
    
    clean_catalogue = movie_catalogue.copy()
    
    clean_catalogue["genres"] = clean_catalogue["genres"].replace("(no genres listed)" , pd.NA)
    
    return clean_catalogue

def extract_title_features(movie_catalogue : pd.DataFrame) -> pd.DataFrame :
    """Create separate title and release-year features."""
    
    catalogue_with_features = movie_catalogue.copy()
    catalogue_with_features["release_year"] = (
        catalogue_with_features["title"].str.extract(r"\((\d{4})\)$", expand=False)
        .astype("Int64")
    )
    
    catalogue_with_features["clean_title"] = (
        catalogue_with_features["title"]
        .str.replace(r"\s*\(\d{4}\)$", "", regex=True)
        .str.strip()
    )

    return catalogue_with_features

def create_genre_features(movie_catalogue : pd.DataFrame) -> pd.DataFrame:
    """Create model-friendly genre features."""
    
    catalogue_with_genres = movie_catalogue.copy()
    catalogue_with_genres["genres_text"] = (
        catalogue_with_genres["genres"]
        .fillna("")
        .str.replace("|" , " ", regex=False)
    )
    
    catalogue_with_genres["genre_count"] = (
        catalogue_with_genres["genres"]
        .apply(lambda genres:0 if pd.isna(genres) else len(genres.split("|")))
    )
    
    return catalogue_with_genres

def save_processed_catalogue(movie_catalogue : pd.DataFrame , output_path : Path) -> None :
    """Save the processed movie catalogue as a CSV file."""
    
    output_path.parent.mkdir(parents=True , exist_ok=True)
    movie_catalogue.to_csv(output_path , index=False)
    
    print(f"\nProcessed catalogue saved to: {output_path}")

if __name__ == "__main__" :
    
    movies , links = load_raw_data()
    
    inspect_movies(movies)
    
    inspect_genres(movies)
    
    print(f"Movies shape: {movies.shape}")
    print(f"Links shape: {links.shape}")
    print("\nMovies columns:")
    print(movies.columns.tolist())
    
    movie_catalogue = merge_tmdb_ids(movies , links) 
    
    print(f"\nCatalogue shape after merge: {movie_catalogue.shape}")
    print(f"Missing TMDB IDs: {movie_catalogue['tmdbId'].isna().sum()}")
    
    clean_catalogue = clean_genres(movie_catalogue)
    print(f"Missing genres after cleaning: {clean_catalogue['genres'].isna().sum()}")
    
    catalogue_with_title_features = extract_title_features(clean_catalogue)

    print("\nTitle feature examples:")
    print(
        catalogue_with_title_features[
            ["title", "clean_title", "release_year"]
        ].head()
    )

    print(
        f"Missing release years: "
        f"{catalogue_with_title_features['release_year'].isna().sum()}"
    )
    
    final_catalogue = create_genre_features(catalogue_with_title_features)

    print("\nGenre feature examples:")
    print(
        final_catalogue[
            ["clean_title", "genres", "genres_text", "genre_count"]
        ].head()
    )
    
    save_processed_catalogue(final_catalogue, PROCESSED_MOVIES_PATH)