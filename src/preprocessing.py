import os
import pickle
import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SIMILARITY_PKL_PATH = os.path.join(DATA_DIR, "similarity.pkl")
VECTORS_PKL_PATH = os.path.join(DATA_DIR, "vectors.pkl")
SVD_MODEL_PATH = os.path.join(DATA_DIR, "svd_model.pkl")

stemmer = PorterStemmer()

def stem_text(text):
    """
    Applies Porter Stemmer on string as described in presentation Slide 9.
    E.g., 'adventures', 'adventure' -> 'adventur'
    """
    if not isinstance(text, str):
        return ""
    words = []
    for word in text.split():
        words.append(stemmer.stem(word))
    return " ".join(words)

def compute_weighted_ratings(df, m_quantile=0.60):
    """
    Computes IMDb/TMDB Bayesian Weighted Rating (WR):
    WR = (v / (v + m)) * R + (m / (v + m)) * C
    where:
    - v: number of votes for the movie
    - m: minimum votes required to be listed
    - R: average rating of the movie
    - C: mean vote across the whole report
    """
    v = df["vote_count"].values
    R = df["vote_average"].values
    C = np.mean(R)
    m = df["vote_count"].quantile(m_quantile)
    
    # Avoid division by zero
    denom = v + m
    denom = np.where(denom == 0, 1e-6, denom)
    wr = (v / denom) * R + (m / denom) * C
    return wr

def build_models(df, max_features=5000, svd_components=50, force_recompute=False):
    """
    Builds:
    1. CountVectorizer / TF-IDF Vector matrix
    2. Cosine Similarity Matrix (4806 x 4806)
    3. Latent SVD Matrix Factorization representations
    4. IMDb Bayesian Weighted Ratings
    """
    if not force_recompute and os.path.exists(SIMILARITY_PKL_PATH) and os.path.exists(VECTORS_PKL_PATH):
        try:
            with open(SIMILARITY_PKL_PATH, "rb") as f:
                similarity = pickle.load(f)
            with open(VECTORS_PKL_PATH, "rb") as f:
                vectors_data = pickle.load(f)
            print("Loaded cached similarity matrix and vectors.")
            return similarity, vectors_data
        except Exception as e:
            print(f"Error loading cached similarity: {e}. Recomputing...")

    print("Stemming movie tags using Porter Stemmer...")
    df["tags_stemmed"] = df["tags"].apply(stem_text)

    print(f"Fitting CountVectorizer (max_features={max_features})...")
    cv = CountVectorizer(max_features=max_features, stop_words="english")
    count_vectors = cv.fit_transform(df["tags_stemmed"]).toarray()

    print(f"Fitting TF-IDF Vectorizer (max_features={max_features})...")
    tfidf = TfidfVectorizer(max_features=max_features, stop_words="english")
    tfidf_vectors = tfidf.fit_transform(df["tags_stemmed"]).toarray()

    print(f"Computing Latent SVD Matrix Factorization (components={svd_components})...")
    svd = TruncatedSVD(n_components=svd_components, random_state=42)
    svd_latent_vectors = svd.fit_transform(tfidf_vectors)

    print("Computing Cosine Similarity matrix...")
    # Calculate similarity using count_vectors or tfidf_vectors (using count_vectors as per PPT slide 5)
    similarity = cosine_similarity(count_vectors).astype(np.float32)

    # Compute Bayesian weighted ratings
    df["weighted_rating"] = compute_weighted_ratings(df)

    vectors_data = {
        "count_vectors": count_vectors,
        "tfidf_vectors": tfidf_vectors,
        "svd_latent_vectors": svd_latent_vectors,
        "feature_names": cv.get_feature_names_out(),
        "svd_model": svd
    }

    try:
        with open(SIMILARITY_PKL_PATH, "wb") as f:
            pickle.dump(similarity, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(VECTORS_PKL_PATH, "wb") as f:
            pickle.dump(vectors_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Successfully cached similarity and vectors data.")
    except Exception as e:
        print(f"Warning: Could not cache similarity data: {e}")

    return similarity, vectors_data

if __name__ == "__main__":
    from data_loader import load_and_preprocess_dataset
    df = load_and_preprocess_dataset()
    similarity, vectors = build_models(df)
    print("Models built successfully!")
    print(f"Similarity matrix shape: {similarity.shape}")
