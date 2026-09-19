import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from poster_fetcher import fetch_poster, get_trailer_url

class MovieRecommender:
    def __init__(self, df, similarity_matrix, vectors_data=None):
        self.df = df.reset_index(drop=True)
        self.similarity_matrix = similarity_matrix
        self.vectors_data = vectors_data or {}
        
        # Build fast lookup dictionary: title.lower() -> index
        self.title_to_idx = {title.strip().lower(): idx for idx, title in enumerate(self.df["title"])}
        
        # Calculate normalized weighted rating (0 to 1 scale) for hybrid scoring
        if "weighted_rating" not in self.df.columns:
            from preprocessing import compute_weighted_ratings
            self.df["weighted_rating"] = compute_weighted_ratings(self.df)
            
        min_wr = self.df["weighted_rating"].min()
        max_wr = self.df["weighted_rating"].max()
        if max_wr > min_wr:
            self.df["norm_rating"] = (self.df["weighted_rating"] - min_wr) / (max_wr - min_wr)
        else:
            self.df["norm_rating"] = 0.5

    def get_all_titles(self):
        return sorted(self.df["title"].dropna().unique().tolist())

    def get_all_genres(self):
        genres = set()
        for g_list in self.df["genres_list"]:
            if isinstance(g_list, list):
                genres.update(g_list)
        return sorted(list(genres))

    def get_movie_details(self, idx):
        row = self.df.iloc[idx]
        movie_id = row.get("movie_id", idx)
        poster_path = row.get("poster_path", None)
        return {
            "idx": idx,
            "movie_id": movie_id,
            "title": row.get("title", "Unknown"),
            "overview": row.get("overview", "No plot description available."),
            "genres": row.get("genres_list", []),
            "cast": row.get("cast_list", []),
            "director": row.get("director", "Unknown Director"),
            "vote_average": round(float(row.get("vote_average", 0.0)), 1),
            "vote_count": int(row.get("vote_count", 0)),
            "release_year": int(row.get("release_year", 2000)),
            "popularity": round(float(row.get("popularity", 0.0)), 1),
            "poster_url": fetch_poster(movie_id, poster_path),
            "trailer_url": get_trailer_url(row.get("title", ""))
        }

    def recommend(
        self,
        movie_title,
        top_k=5,
        genre_filter=None,
        min_rating=0.0,
        year_range=(1900, 2030),
        alpha_content=0.75, # Weight for content similarity vs. rating quality (Hybrid)
        sort_by="hybrid" # 'hybrid', 'similarity', 'rating', 'popularity', 'newest'
    ):
        clean_title = movie_title.strip().lower()
        if clean_title not in self.title_to_idx:
            # Try fuzzy match if exact match fails
            matches = [t for t in self.title_to_idx.keys() if clean_title in t]
            if matches:
                target_idx = self.title_to_idx[matches[0]]
            else:
                return []
        else:
            target_idx = self.title_to_idx[clean_title]

        # Get cosine similarity array for target movie
        sim_scores = self.similarity_matrix[target_idx]

        # If SVD latent vectors exist, blend in latent semantic similarity
        if "svd_latent_vectors" in self.vectors_data:
            svd_vecs = self.vectors_data["svd_latent_vectors"]
            target_svd = svd_vecs[target_idx].reshape(1, -1)
            svd_sims = cosine_similarity(target_svd, svd_vecs)[0]
            # Blend CountVectorizer cosine and SVD latent cosine
            content_sim_scores = 0.70 * sim_scores + 0.30 * svd_sims
        else:
            content_sim_scores = sim_scores

        candidates = []
        for idx in range(len(self.df)):
            if idx == target_idx:
                continue

            row = self.df.iloc[idx]
            
            # Apply Filters
            # 1. Rating Filter
            if row.get("vote_average", 0) < min_rating:
                continue
            # 2. Release Year Filter
            rel_year = row.get("release_year", 2000)
            if not (year_range[0] <= rel_year <= year_range[1]):
                continue
            # 3. Genre Filter (match any selected genre if provided)
            if genre_filter and len(genre_filter) > 0:
                movie_genres = row.get("genres_list", [])
                if not any(g in movie_genres for g in genre_filter):
                    continue

            content_score = float(content_sim_scores[idx])
            rating_score = float(row.get("norm_rating", 0.5))
            hybrid_score = (alpha_content * content_score) + ((1.0 - alpha_content) * rating_score)

            candidates.append({
                "idx": idx,
                "content_score": content_score,
                "rating_score": rating_score,
                "hybrid_score": hybrid_score,
                "vote_average": row.get("vote_average", 0.0),
                "popularity": row.get("popularity", 0.0),
                "release_year": rel_year
            })

        if not candidates:
            return []

        # Sort candidates
        if sort_by == "similarity":
            candidates.sort(key=lambda x: x["content_score"], reverse=True)
        elif sort_by == "rating":
            candidates.sort(key=lambda x: x["vote_average"], reverse=True)
        elif sort_by == "popularity":
            candidates.sort(key=lambda x: x["popularity"], reverse=True)
        elif sort_by == "newest":
            candidates.sort(key=lambda x: x["release_year"], reverse=True)
        else: # 'hybrid' default
            candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)

        # Build detailed result list
        results = []
        for item in candidates[:top_k]:
            details = self.get_movie_details(item["idx"])
            details["similarity_score"] = round(item["content_score"] * 100, 1)
            details["hybrid_score"] = round(item["hybrid_score"] * 100, 1)
            results.append(details)

        return results

    def discover_by_genres(self, genres, top_k=10, min_rating=6.0, year_range=(1900, 2030), sort_by="rating"):
        """Browse movies when filtering by genres without a specific seed movie."""
        filtered_indices = []
        for idx, row in self.df.iterrows():
            if row.get("vote_average", 0) < min_rating:
                continue
            rel_year = row.get("release_year", 2000)
            if not (year_range[0] <= rel_year <= year_range[1]):
                continue
            movie_genres = row.get("genres_list", [])
            if genres and not any(g in movie_genres for g in genres):
                continue
            filtered_indices.append(idx)

        if not filtered_indices:
            return []

        sub_df = self.df.iloc[filtered_indices].copy()
        if sort_by == "popularity":
            sub_df = sub_df.sort_values(by="popularity", ascending=False)
        elif sort_by == "newest":
            sub_df = sub_df.sort_values(by="release_year", ascending=False)
        else:
            sub_df = sub_df.sort_values(by="vote_average", ascending=False)

        results = []
        for idx in sub_df.index[:top_k]:
            details = self.get_movie_details(idx)
            details["similarity_score"] = 100.0
            details["hybrid_score"] = 100.0
            results.append(details)
        return results
