import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="AI Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add src to system path
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_loader import load_and_preprocess_dataset
from preprocessing import build_models, stem_text
from recommender import MovieRecommender
from poster_fetcher import fetch_poster, get_trailer_url
from visualization import generate_pca_comparison_plot

# Custom High-End Styling (Netflix / Modern Cinematic Dark Theme)
st.markdown("""
<style>
    /* Dark Cinematic Theme */
    .stApp {
        background-color: #0e1117;
        color: #f0f2f6;
    }
    
    /* Main Hero Title */
    .hero-container {
        background: linear-gradient(135deg, rgba(229, 9, 20, 0.25) 0%, rgba(20, 20, 20, 0.95) 100%);
        border-left: 6px solid #E50914;
        border-radius: 12px;
        padding: 24px 30px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #FFFFFF;
        margin: 0;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #B3B3B3;
        margin-top: 8px;
    }
    
    /* Metrics Row */
    .metric-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 5px 14px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #E5E5E5;
        margin-right: 10px;
        margin-top: 10px;
    }
    
    /* Movie Card Styling */
    .movie-card {
        background: #181c24;
        border-radius: 14px;
        border: 1px solid #2a2f3d;
        overflow: hidden;
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        margin-bottom: 20px;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 28px rgba(229, 9, 20, 0.22);
        border-color: #E50914;
    }
    
    .poster-img {
        width: 100%;
        height: 310px;
        object-fit: cover;
        border-top-left-radius: 14px;
        border-top-right-radius: 14px;
    }
    
    .card-body {
        padding: 14px 16px;
        display: flex;
        flex-direction: column;
        flex-grow: 1;
    }
    
    .card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 6px;
        line-height: 1.3;
        height: 2.6rem;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }
    
    .badge-pill {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 6px;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .badge-match {
        background-color: #E50914;
        color: white;
    }
    .badge-rating {
        background-color: #FFB800;
        color: #111111;
    }
    .badge-year {
        background-color: #2b3548;
        color: #cbd5e1;
    }
    
    .genre-tag {
        display: inline-block;
        background-color: #1f2736;
        color: #94a3b8;
        font-size: 0.72rem;
        padding: 2px 7px;
        border-radius: 4px;
        margin-right: 4px;
        margin-bottom: 4px;
    }

    .meta-text {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 4px;
        margin-bottom: 2px;
    }

    /* Target Movie Spotlight Card */
    .spotlight-card {
        background: linear-gradient(145deg, #1c2230 0%, #131722 100%);
        border: 1px solid #3b82f6;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 25px;
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.15);
    }

    /* Button Customization */
    .stButton>button {
        background: linear-gradient(90deg, #E50914 0%, #b80710 100%);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 8px 18px;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #ff1a26 0%, #d40812 100%);
        box-shadow: 0 4px 15px rgba(229, 9, 20, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading & model building
@st.cache_resource(show_spinner="🎬 Initializing AI Recommendation Engine & TMDB Dataset...")
def get_engine():
    df = load_and_preprocess_dataset()
    similarity, vectors_data = build_models(df)
    recommender = MovieRecommender(df, similarity, vectors_data)
    return recommender, df, vectors_data

try:
    recommender, df, vectors_data = get_engine()
except Exception as e:
    st.error(f"Failed to initialize recommender engine: {e}")
    st.stop()

# Header Banner
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🎬 AI Movie Recommendation System</div>
    <div class="hero-subtitle">Hybrid Machine Learning Engine combining TF-IDF Content Similarity, Porter Stemming, and Latent Factor SVD Matrix Factorization</div>
    <div>
        <span class="metric-badge">📚 4,800+ TMDB Movies</span>
        <span class="metric-badge">🧠 5,000D Feature Space</span>
        <span class="metric-badge">⚙️ Porter Stemmer & Cosine Sim</span>
        <span class="metric-badge">🎯 SVD Matrix Factorization</span>
        <span class="metric-badge">📊 2D PCA Visualizer</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab_hybrid, tab_discover, tab_pca, tab_arch = st.tabs([
    "🎯 Hybrid AI Recommender",
    "🔍 Genre & Mood Explorer",
    "🌌 2D PCA Feature Visualizer",
    "🧠 PEAS & ML Architecture"
])

# ==============================================================================
# TAB 1: HYBRID AI RECOMMENDER
# ==============================================================================
with tab_hybrid:
    col_input, col_config = st.columns([1.5, 1])
    
    all_titles = recommender.get_all_titles()
    all_genres = recommender.get_all_genres()
    
    with col_input:
        default_index = all_titles.index("The Dark Knight Rises") if "The Dark Knight Rises" in all_titles else 0
        
        # Check if a movie was selected via quick-action in session state
        if "selected_movie" in st.session_state and st.session_state["selected_movie"] in all_titles:
            default_index = all_titles.index(st.session_state["selected_movie"])
            
        selected_movie = st.selectbox(
            "🎥 Select or Type a Movie You Enjoy:",
            options=all_titles,
            index=default_index,
            help="Choose a movie to find personalized recommendations matching its plot, genre, cast, and director."
        )

        # Quick Picks Row
        st.markdown("<div style='font-size: 0.85rem; color: #94a3b8; margin-top: 4px;'>🔥 Quick Picks:</div>", unsafe_allow_html=True)
        qp_cols = st.columns(6)
        popular_picks = ["Avatar", "Inception", "The Avengers", "Interstellar", "Pulp Fiction", "The Matrix"]
        for idx, p_title in enumerate(popular_picks):
            if p_title in all_titles:
                with qp_cols[idx]:
                    if st.button(p_title, key=f"qp_{idx}", use_container_width=True):
                        st.session_state["selected_movie"] = p_title
                        st.rerun()

    with col_config:
        st.markdown("<div style='font-weight: 600; color: #cbd5e1; margin-bottom: 6px;'>⚙️ Recommendation Filters:</div>", unsafe_allow_html=True)
        
        # Genre multi-select filter
        genre_filter = st.multiselect(
            "🎭 Filter by Specific Genres (Action, Drama, Thriller, etc.):",
            options=all_genres,
            default=[],
            help="Leave empty to search all genres, or select one or more genres to narrow recommendations."
        )
        
        sub_c1, sub_c2 = st.columns(2)
        with sub_c1:
            top_k = st.slider("🔢 Number of Movies (Top-K):", min_value=3, max_value=12, value=5, step=1)
            min_rating = st.slider("⭐ Minimum Rating:", min_value=0.0, max_value=9.0, value=5.0, step=0.5)
        with sub_c2:
            sort_option = st.selectbox(
                "📊 Ranking Strategy:",
                options=["hybrid", "similarity", "rating", "popularity", "newest"],
                format_func=lambda x: {
                    "hybrid": "🎯 AI Hybrid Score",
                    "similarity": "🧬 Pure Content Match",
                    "rating": "⭐ Highest Rated",
                    "popularity": "🔥 Most Popular",
                    "newest": "📅 Newest Release"
                }[x]
            )
            alpha_content = st.slider(
                "🎛️ Hybrid Weight (Content vs. Ratings):",
                min_value=0.0,
                max_value=1.0,
                value=0.75,
                step=0.05,
                help="1.0 = Pure plot/cast similarity, 0.0 = Pure ratings & popularity quality."
            )

    # Display Selected Target Movie Spotlight
    if selected_movie:
        clean_title = selected_movie.strip().lower()
        if clean_title in recommender.title_to_idx:
            target_idx = recommender.title_to_idx[clean_title]
            target_details = recommender.get_movie_details(target_idx)
            
            with st.expander("📌 View Selected Movie Profile", expanded=False):
                sc1, sc2 = st.columns([1, 4])
                with sc1:
                    st.image(target_details["poster_url"], width=160)
                with sc2:
                    st.markdown(f"### {target_details['title']} ({target_details['release_year']})")
                    st.markdown(f"⭐ **Rating:** {target_details['vote_average']}/10 ({target_details['vote_count']:,} votes) | 🎬 **Director:** {target_details['director']}")
                    st.markdown(f"🎭 **Genres:** {', '.join(target_details['genres'])} | 👥 **Top Cast:** {', '.join(target_details['cast'])}")
                    st.markdown(f"📖 **Overview:** {target_details['overview']}")
                    st.markdown(f"[▶️ Watch Official Trailer on YouTube]({target_details['trailer_url']})")

    st.markdown("---")
    
    # Generate Recommendations
    with st.spinner("🧠 Computing similarity vectors and matrix rankings..."):
        recs = recommender.recommend(
            movie_title=selected_movie,
            top_k=top_k,
            genre_filter=genre_filter,
            min_rating=min_rating,
            alpha_content=alpha_content,
            sort_by=sort_option
        )

    if not recs:
        st.warning("⚠️ No movies matched your selected filter criteria. Try lowering the minimum rating or clearing genre filters.")
    else:
        st.markdown(f"### 🍿 Top {len(recs)} Recommended Movies for *{selected_movie}*")
        
        # Display recommendations in responsive columns
        n_cols = min(len(recs), 5)
        grid_cols = st.columns(n_cols)
        
        for i, movie in enumerate(recs):
            col_idx = i % n_cols
            with grid_cols[col_idx]:
                # Movie Card HTML
                genres_html = "".join([f"<span class='genre-tag'>{g}</span>" for g in movie['genres'][:3]])
                card_html = f"""
                <div class="movie-card">
                    <img src="{movie['poster_url']}" class="poster-img" alt="{movie['title']}">
                    <div class="card-body">
                        <div class="card-title" title="{movie['title']}">{movie['title']}</div>
                        <div>
                            <span class="badge-pill badge-match">{movie['similarity_score']}% Match</span>
                            <span class="badge-pill badge-rating">⭐ {movie['vote_average']}</span>
                            <span class="badge-pill badge-year">{movie['release_year']}</span>
                        </div>
                        <div style="margin-top: 6px;">
                            {genres_html}
                        </div>
                        <div class="meta-text">🎬 <b>Dir:</b> {movie['director']}</div>
                        <div class="meta-text">👥 <b>Cast:</b> {', '.join(movie['cast'][:2])}</div>
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
                
                # Interactive Action Buttons
                b_col1, b_col2 = st.columns(2)
                with b_col1:
                    st.markdown(f"<a href='{movie['trailer_url']}' target='_blank' style='text-decoration:none;'><button style='width:100%; background:#252d3d; color:#e2e8f0; border:1px solid #3b4252; padding:5px 8px; border-radius:6px; font-size:0.78rem; cursor:pointer;'>▶️ Trailer</button></a>", unsafe_allow_html=True)
                with b_col2:
                    if st.button("🔄 Similar", key=f"sim_btn_{movie['idx']}", use_container_width=True):
                        st.session_state["selected_movie"] = movie["title"]
                        st.rerun()

                with st.expander("📖 Storyline", expanded=False):
                    st.write(movie["overview"])

# ==============================================================================
# TAB 2: GENRE & MOOD EXPLORER
# ==============================================================================
with tab_discover:
    st.markdown("### 🎭 Explore by Genre & Mood")
    st.markdown("Discover the highest-rated and trending movies across specific genres without needing a seed movie.")
    
    d_col1, d_col2, d_col3, d_col4 = st.columns([2, 1, 1, 1])
    
    with d_col1:
        disc_genres = st.multiselect(
            "Select Desired Genres:",
            options=all_genres,
            default=["Action", "Thriller"] if "Action" in all_genres and "Thriller" in all_genres else [all_genres[0]],
            key="disc_genres"
        )
    with d_col2:
        disc_rating = st.slider("Minimum Rating:", 0.0, 9.0, 6.5, 0.5, key="disc_rating")
    with d_col3:
        disc_year_range = st.slider("Release Year Range:", 1970, 2026, (2000, 2026), key="disc_years")
    with d_col4:
        disc_sort = st.selectbox("Sort By:", ["rating", "popularity", "newest"], format_func=lambda x: {"rating": "⭐ Highest Rating", "popularity": "🔥 Most Popular", "newest": "📅 Newest Release"}[x], key="disc_sort")
        
    discovered_movies = recommender.discover_by_genres(
        genres=disc_genres,
        top_k=10,
        min_rating=disc_rating,
        year_range=disc_year_range,
        sort_by=disc_sort
    )
    
    if not discovered_movies:
        st.info("No movies found matching these filter criteria. Try widening the year range or lowering the rating.")
    else:
        st.markdown(f"#### Found {len(discovered_movies)} Matching Titles:")
        d_grid = st.columns(5)
        for i, movie in enumerate(discovered_movies):
            col_idx = i % 5
            with d_grid[col_idx]:
                genres_html = "".join([f"<span class='genre-tag'>{g}</span>" for g in movie['genres'][:3]])
                card_html = f"""
                <div class="movie-card">
                    <img src="{movie['poster_url']}" class="poster-img" alt="{movie['title']}">
                    <div class="card-body">
                        <div class="card-title" title="{movie['title']}">{movie['title']}</div>
                        <div>
                            <span class="badge-pill badge-rating">⭐ {movie['vote_average']}</span>
                            <span class="badge-pill badge-year">{movie['release_year']}</span>
                        </div>
                        <div style="margin-top: 6px;">
                            {genres_html}
                        </div>
                        <div class="meta-text">🎬 <b>Dir:</b> {movie['director']}</div>
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
                if st.button("🎯 Recommend Similar", key=f"disc_rec_{movie['idx']}", use_container_width=True):
                    st.session_state["selected_movie"] = movie["title"]
                    st.rerun()

# ==============================================================================
# TAB 3: 2D PCA FEATURE SPACE VISUALIZER
# ==============================================================================
with tab_pca:
    st.markdown("### 🌌 2D PCA Latent Semantic Space Visualizer")
    st.markdown("""
    This interactive visualizer uses **Principal Component Analysis (PCA)** to project high-dimensional TF-IDF vector embeddings 
    into a 2D coordinate plane. **Movies that share similar plots, genres, cast, and directors cluster closer together.**
    *(Direct interactive recreation of Presentation Slides 7 & 8)*
    """)
    
    pca_presets = {
        "🦸 Avengers Franchise (Slide 7)": [
            "The Avengers", "Avengers: Age of Ultron", "Avengers: Infinity War", "Captain America: Civil War", "Iron Man 3"
        ],
        "🦇 Batman / DC Universe (Slide 8)": [
            "The Dark Knight Rises", "Batman Begins", "Batman v Superman: Dawn of Justice", "Justice League", "Batman & Robin", "Batman Forever"
        ],
        "⚡ Harry Potter Universe": [
            "Harry Potter and the Philosopher's Stone", "Harry Potter and the Chamber of Secrets", "Harry Potter and the Prisoner of Azkaban", "Harry Potter and the Goblet of Fire", "Harry Potter and the Deathly Hallows: Part 2"
        ],
        "🌌 Sci-Fi Masterpieces": [
            "Interstellar", "Inception", "The Matrix", "Avatar", "Gravity", "The Martian", "Blade Runner"
        ],
        "🎥 Christopher Nolan Films": [
            "Inception", "The Dark Knight", "The Dark Knight Rises", "Batman Begins", "Interstellar", "Memento", "The Prestige"
        ],
        "🎨 Custom Selection": []
    }
    
    preset_choice = st.selectbox("Choose a Franchise Preset or Custom Set:", list(pca_presets.keys()))
    
    if preset_choice == "🎨 Custom Selection":
        custom_movies = st.multiselect(
            "Select 3 to 10 movies to project in 2D space:",
            options=all_titles,
            default=["Avatar", "Titanic", "Inception", "The Matrix", "The Dark Knight Rises"] if all(x in all_titles for x in ["Avatar", "Titanic", "Inception"]) else all_titles[:5]
        )
        selected_pca_movies = custom_movies
    else:
        selected_pca_movies = pca_presets[preset_choice]
        st.info(f"Showing movies: {', '.join(selected_pca_movies)}")
        
    fig, err = generate_pca_comparison_plot(
        df,
        vectors_data,
        selected_pca_movies,
        title_label=preset_choice
    )
    
    if err:
        st.warning(err)
    elif fig:
        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 **Interpretation**: Points situated closely in 2D PCA space possess higher cosine similarity across their overview plot, genre tags, director, and cast.")

# ==============================================================================
# TAB 4: PEAS & ML ARCHITECTURE EXPLORER
# ==============================================================================
with tab_arch:
    st.markdown("### 🧠 AI System Architecture & PEAS Agent Model")
    
    # PEAS Model Cards (Slide 6)
    st.markdown("#### 🤖 PEAS Agent Specification (Slide 6)")
    p_c1, p_c2, p_c3, p_c4 = st.columns(4)
    with p_c1:
        st.markdown("""
        <div style="background:#1e293b; padding:16px; border-radius:10px; border-left:4px solid #38bdf8;">
            <h4 style="color:#38bdf8; margin:0 0 8px 0;">Performance (P)</h4>
            <p style="font-size:0.85rem; color:#cbd5e1; margin:0;">
                Accuracy & relevance of recommendations, Cosine similarity score, Hybrid Top-K ranking precision.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with p_c2:
        st.markdown("""
        <div style="background:#1e293b; padding:16px; border-radius:10px; border-left:4px solid #34d399;">
            <h4 style="color:#34d399; margin:0 0 8px 0;">Environment (E)</h4>
            <p style="font-size:0.85rem; color:#cbd5e1; margin:0;">
                TMDB Movie Dataset containing 4,800+ movies, plot overviews, genres, cast, crew, ratings, and popularity metrics.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with p_c3:
        st.markdown("""
        <div style="background:#1e293b; padding:16px; border-radius:10px; border-left:4px solid #f43f5e;">
            <h4 style="color:#f43f5e; margin:0 0 8px 0;">Actuators (A)</h4>
            <p style="font-size:0.85rem; color:#cbd5e1; margin:0;">
                Streamlit interactive UI rendering dynamic movie cards, poster thumbnails, similarity match %, and 2D PCA charts.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with p_c4:
        st.markdown("""
        <div style="background:#1e293b; padding:16px; border-radius:10px; border-left:4px solid #fbbf24;">
            <h4 style="color:#fbbf24; margin:0 0 8px 0;">Sensors (S)</h4>
            <p style="font-size:0.85rem; color:#cbd5e1; margin:0;">
                User search bar, genre multi-select filters, Top-K sliders, rating thresholds, and hybrid weight controls.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Text Processing & Porter Stemmer Sandbox (Slide 9)
    st.markdown("#### 🔤 Text Processing: Porter Stemmer Sandbox (Slide 9)")
    st.markdown("The Porter Stemmer strips suffixes and reduces words to their root morphological base so synonyms and inflections match accurately.")
    
    sample_text = st.text_input("Test Stemmer with Custom Text:", value="Action Adventures Fantasies directors acting directing")
    if sample_text:
        stemmed_res = stem_text(sample_text)
        s_c1, s_c2 = st.columns(2)
        with s_c1:
            st.code(f"Original: {sample_text.split()}", language="python")
        with s_c2:
            st.code(f"Stemmed:  {stemmed_res.split()}", language="python")

    st.markdown("---")
    
    # Algorithm Workflow Breakdown (Slides 3, 4, 5)
    st.markdown("#### 📐 Algorithm & Math Pipeline")
    m_c1, m_c2 = st.columns(2)
    with m_c1:
        st.markdown(r"""
        **1. Feature Engineering & Tag Consolidation:**
        Combining 5 core features:
        $$\text{Tags} = \text{Overview} + \text{Genres} + \text{Keywords} + \text{Top 3 Cast} + \text{Director}$$
        
        **2. Vectorization:**
        Transforming text tags into 5,000-dimensional sparse vectors using CountVectorizer & TF-IDF:
        $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{|D|}{1 + |\{d \in D : t \in d\}|}\right)$$
        """)
    with m_c2:
        st.markdown(r"""
        **3. Cosine Similarity Scoring:**
        $$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} = \frac{\sum_{i=1}^{n} u_i v_i}{\sqrt{\sum_{i=1}^n u_i^2} \sqrt{\sum_{i=1}^n v_i^2}}$$
        
        **4. Hybrid Bayesian Ranking:**
        $$\text{Weighted Rating (WR)} = \left(\frac{v}{v + m}\right) R + \left(\frac{m}{v + m}\right) C$$
        $$\text{Final Hybrid Score} = \alpha \cdot \text{Sim}_{\text{Content}} + (1 - \alpha) \cdot \text{WR}_{\text{Norm}}$$
        """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#94a3b8; font-size:0.85rem; padding:10px;">
        👥 <b>Project Team:</b> Sanjay Gedela (23BKT0139) | Ishan Singh (23BCI0104) | Khushi Singh (21BCT0364) | Shrutika (23BCE0116) | Jain Utkarsh Sandeep (23BCT0088)
    </div>
    """, unsafe_allow_html=True)
