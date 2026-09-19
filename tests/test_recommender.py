import os
import sys

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_loader import load_and_preprocess_dataset
from preprocessing import build_models
from recommender import MovieRecommender
from visualization import generate_pca_comparison_plot
from poster_fetcher import fetch_poster, get_trailer_url

def test_full_pipeline():
    print("Testing Pipeline...")
    df = load_and_preprocess_dataset()
    assert len(df) > 4000, f"Expected >4000 movies, got {len(df)}"
    print(f"[PASS] Dataset loaded: {len(df)} movies")

    similarity, vectors_data = build_models(df)
    assert similarity.shape == (len(df), len(df)), f"Unexpected similarity shape: {similarity.shape}"
    print(f"[PASS] Similarity matrix verified: {similarity.shape}")

    recommender = MovieRecommender(df, similarity, vectors_data)
    
    # Test 1: Recommend for The Dark Knight Rises
    recs = recommender.recommend("The Dark Knight Rises", top_k=5)
    assert len(recs) == 5, f"Expected 5 recommendations, got {len(recs)}"
    print(f"[PASS] 5 Recommendations for 'The Dark Knight Rises':")
    for r in recs:
        print(f"   - {r['title']} ({r['release_year']}) | Match: {r['similarity_score']}% | Rating: {r['vote_average']}")

    # Test 2: Recommend with Genre Filter (Action + Drama)
    action_recs = recommender.recommend("The Dark Knight Rises", top_k=5, genre_filter=["Action", "Drama"])
    print(f"[PASS] Recommendations with Genre Filter (Action, Drama):")
    for r in action_recs:
        print(f"   - {r['title']} | Genres: {r['genres']}")
        assert any(g in ["Action", "Drama"] for g in r["genres"]), "Filter constraint failed"

    # Test 3: Discover by Genres
    discovered = recommender.discover_by_genres(["Thriller", "Crime"], top_k=3, min_rating=7.0)
    assert len(discovered) > 0, "Expected discovered movies"
    print(f"[PASS] Genre Explorer (Thriller, Crime, min_rating>=7.0):")
    for d in discovered:
        print(f"   - {d['title']} | Rating: {d['vote_average']} | Genres: {d['genres']}")

    # Test 4: PCA 2D Plot Generation
    fig, err = generate_pca_comparison_plot(
        df,
        vectors_data,
        ["The Dark Knight Rises", "Batman Begins", "Batman v Superman: Dawn of Justice", "Justice League"],
        title_label="Batman Comparison"
    )
    assert err is None and fig is not None, f"PCA plotting error: {err}"
    print("[PASS] PCA 2D scatter plot generated successfully")

    # Test 5: Poster and Trailer URL
    trailer = get_trailer_url("The Dark Knight Rises")
    poster = fetch_poster(49026)
    assert "youtube" in trailer, "Invalid trailer url"
    assert poster.startswith("http"), "Invalid poster url"
    print(f"[PASS] Poster fetcher & Trailer URLs verified: {poster[:40]}... | {trailer}")

    print("\n[SUCCESS] ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_full_pipeline()
