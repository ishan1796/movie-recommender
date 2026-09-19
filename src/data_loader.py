import os
import ast
import json
import urllib.request
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MOVIES_CSV_PATH = os.path.join(DATA_DIR, "tmdb_5000_movies.csv")
CREDITS_CSV_PATH = os.path.join(DATA_DIR, "tmdb_5000_credits.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "movies_processed.pkl")

# Public mirror URLs for TMDB 5000 / 10000 dataset
MOVIES_URLS = [
    "https://raw.githubusercontent.com/redwankarimsony/TMDB-5000-Movie-Dataset/master/tmdb_5000_movies.csv",
    "https://raw.githubusercontent.com/shantanuo/kaggle-tmdb-5000-movies/master/tmdb_5000_movies.csv",
    "https://raw.githubusercontent.com/campusx-official/movie-recommender-system-tmdb-dataset/main/tmdb_5000_movies.csv"
]

CREDITS_URLS = [
    "https://raw.githubusercontent.com/redwankarimsony/TMDB-5000-Movie-Dataset/master/tmdb_5000_credits.csv",
    "https://raw.githubusercontent.com/shantanuo/kaggle-tmdb-5000-movies/master/tmdb_5000_credits.csv",
    "https://raw.githubusercontent.com/campusx-official/movie-recommender-system-tmdb-dataset/main/tmdb_5000_credits.csv"
]

def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def ensure_data_files():
    ensure_data_dir()
    if os.path.exists(MOVIES_CSV_PATH) and os.path.exists(CREDITS_CSV_PATH):
        if os.path.getsize(MOVIES_CSV_PATH) > 10000 and os.path.getsize(CREDITS_CSV_PATH) > 10000:
            return True

    try:
        import kagglehub
        import shutil
        print("Downloading TMDB dataset via kagglehub...", flush=True)
        path = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")
        for item in os.listdir(path):
            src_file = os.path.join(path, item)
            dst_file = os.path.join(DATA_DIR, item)
            shutil.copy(src_file, dst_file)
        print("Successfully acquired TMDB dataset!", flush=True)
        return True
    except Exception as e:
        print(f"kagglehub download error: {e}", flush=True)

    return os.path.exists(MOVIES_CSV_PATH) and os.path.exists(CREDITS_CSV_PATH)

def parse_json_field(text, key="name", limit=None):
    """Safe extraction of names from stringified JSON lists."""
    if not text or pd.isna(text):
        return []
    try:
        items = ast.literal_eval(text)
        if isinstance(items, list):
            names = [item.get(key, "").strip() for item in items if isinstance(item, dict) and key in item]
            names = [n for n in names if n]
            return names[:limit] if limit else names
    except Exception:
        pass
    return []

def get_director(crew_text):
    """Extract director name from crew JSON string."""
    if not crew_text or pd.isna(crew_text):
        return ""
    try:
        crew = ast.literal_eval(crew_text)
        if isinstance(crew, list):
            for member in crew:
                if isinstance(member, dict) and member.get("job") == "Director":
                    return member.get("name", "").strip()
    except Exception:
        pass
    return ""

def load_and_preprocess_dataset(force_recompute=False):
    """
    Loads movies and credits, cleans columns, extracts features:
    - Overview (plot)
    - Genres (list and text)
    - Keywords (list and text)
    - Cast (Top 3 actors)
    - Crew (Director)
    - Ratings, vote counts, popularity, release year
    """
    if not force_recompute and os.path.exists(PROCESSED_DATA_PATH):
        try:
            return pd.read_pickle(PROCESSED_DATA_PATH)
        except Exception as e:
            print(f"Error loading cached pickle: {e}. Recomputing...")

    ensure_data_files()

    if not os.path.exists(MOVIES_CSV_PATH) or not os.path.exists(CREDITS_CSV_PATH):
        raise FileNotFoundError("Could not locate TMDB dataset CSV files in data directory.")

    print("Loading CSV files...")
    movies_df = pd.read_csv(MOVIES_CSV_PATH)
    credits_df = pd.read_csv(CREDITS_CSV_PATH)

    # Merge on title or id
    if "title" in movies_df.columns and "title" in credits_df.columns:
        df = movies_df.merge(credits_df, on="title", suffixes=("", "_credit"))
    elif "id" in movies_df.columns and "movie_id" in credits_df.columns:
        df = movies_df.merge(credits_df, left_on="id", right_on="movie_id")
    else:
        df = movies_df

    # Standardize movie_id
    if "id" in df.columns:
        df["movie_id"] = df["id"]
    elif "movie_id" not in df.columns:
        df["movie_id"] = range(len(df))

    # Keep relevant fields
    df["title"] = df["title"].fillna("Unknown Title")
    df["overview"] = df["overview"].fillna("")
    
    # Extract structured fields
    df["genres_list"] = df["genres"].apply(lambda x: parse_json_field(x, "name"))
    df["keywords_list"] = df["keywords"].apply(lambda x: parse_json_field(x, "name"))
    df["cast_list"] = df["cast"].apply(lambda x: parse_json_field(x, "name", limit=3))
    df["director"] = df["crew"].apply(get_director)
    
    # Numerical / metadata fields
    df["vote_average"] = pd.to_numeric(df.get("vote_average", 0), errors="coerce").fillna(0.0)
    df["vote_count"] = pd.to_numeric(df.get("vote_count", 0), errors="coerce").fillna(0)
    df["popularity"] = pd.to_numeric(df.get("popularity", 0), errors="coerce").fillna(0.0)
    
    # Extract Release Year
    if "release_date" in df.columns:
        df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year.fillna(2000).astype(int)
    else:
        df["release_year"] = 2000

    # Clean strings for tags: remove spaces between first and last names (e.g., 'Christian Bale' -> 'ChristianBale')
    def collapse_names(items):
        return [i.replace(" ", "") for i in items]

    df["genres_clean"] = df["genres_list"].apply(collapse_names)
    df["keywords_clean"] = df["keywords_list"].apply(collapse_names)
    df["cast_clean"] = df["cast_list"].apply(collapse_names)
    df["director_clean"] = df["director"].apply(lambda d: [d.replace(" ", "")] if d else [])

    # Overview into tokens
    df["overview_tokens"] = df["overview"].apply(lambda x: x.split() if isinstance(x, str) else [])

    # Combine all 5 features as per Slide 3 & 4 of PPT
    df["tags_list"] = df["overview_tokens"] + df["genres_clean"] + df["keywords_clean"] + df["cast_clean"] + df["director_clean"]
    df["tags"] = df["tags_list"].apply(lambda x: " ".join(x).lower())

    # Drop duplicate movie titles keeping first
    df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)

    # Save to pickle for super fast subsequent loads
    try:
        df.to_pickle(PROCESSED_DATA_PATH)
        print(f"Saved processed dataset with {len(df)} movies to {PROCESSED_DATA_PATH}")
    except Exception as e:
        print(f"Warning: Could not save pickle: {e}")

    return df

if __name__ == "__main__":
    df = load_and_preprocess_dataset()
    print("Dataset loaded successfully!")
    print(f"Total Movies: {len(df)}")
    print(df[["title", "genres_list", "director", "vote_average", "release_year"]].head())
