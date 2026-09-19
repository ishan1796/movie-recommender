import os
import requests
import functools

# Standard TMDB API Key for educational / demo use; can be overridden via environment or UI
DEFAULT_TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "8265bd1679663a7ea12ac168da84d2e8")
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"
FALLBACK_POSTER = "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?auto=format&fit=crop&w=500&q=80"

# Local cache for fast retrieval
_POSTER_CACHE = {}

def fetch_poster(movie_id, poster_path=None, api_key=None):
    """
    Fetches the poster image URL for a given movie ID using TMDB API.
    If poster_path is provided directly in dataset, uses it directly.
    Falls back gracefully if network fails.
    """
    if movie_id in _POSTER_CACHE:
        return _POSTER_CACHE[movie_id]

    # If poster_path is already a valid relative string (e.g., "/q6y0Go1tsGEsmtFryDOJo3dEmqu.jpg")
    if poster_path and isinstance(poster_path, str) and poster_path.startswith("/"):
        full_url = f"{POSTER_BASE_URL}{poster_path}"
        _POSTER_CACHE[movie_id] = full_url
        return full_url

    key = api_key or DEFAULT_TMDB_API_KEY
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={key}&language=en-US"

    try:
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            data = response.json()
            path = data.get("poster_path")
            if path:
                full_url = f"{POSTER_BASE_URL}{path}"
                _POSTER_CACHE[movie_id] = full_url
                return full_url
    except Exception:
        pass

    # Fallback placeholder
    _POSTER_CACHE[movie_id] = FALLBACK_POSTER
    return FALLBACK_POSTER

def get_trailer_url(title):
    """Returns a direct YouTube search URL for the movie trailer."""
    query = f"{title} official trailer".replace(" ", "+")
    return f"https://www.youtube.com/results?search_query={query}"
