# 🎬 AI Movie Recommendation System

An AI-powered Hybrid Movie Recommendation Platform built with Python, Pandas, NumPy, Scikit-learn, NLTK, Streamlit, and Plotly.

---

## 🌟 Key Features

1. **Hybrid AI Engine**:
   - **Content-Based Filtering**: Preprocessing with **Porter Stemmer**, TF-IDF / CountVectorizer token representation across plot overview, genres, cast, director, and keywords.
   - **Latent SVD Matrix Factorization**: TruncatedSVD dimensionality reduction capturing latent semantic topics and movie relationships.
   - **Bayesian Weighted Rating Integration**: IMDb-weighted rating formula \(WR = \frac{v}{v+m} R + \frac{m}{v+m} C\) balancing popularity vs quality.
   - **Tunable Hybrid Weight**: Real-time slider to shift weighting between pure plot/cast similarity and global rating quality.

2. **Rich Discovery & Multi-Genre Filters**:
   - Filter by genres: Action, Drama, Thriller, Sci-Fi, Adventure, Crime, Comedy, Horror, Fantasy, etc.
   - Filter by Minimum Rating and Release Year.
   - Sorting options: AI Hybrid Score, Content Match %, Highest Rated, Most Popular, Newest.

3. **2D PCA Feature Space Visualizer** *(As showcased in PPT Slides 7 & 8)*:
   - Interactive 2D PCA cluster maps for franchise comparison (Avengers Universe, Batman Universe, Harry Potter, Sci-Fi Masterpieces, Nolan Classics, and Custom Multi-Movie Selection).

4. **Modern Dark Cinematic UI**:
   - Netflix-inspired dark aesthetic with movie poster cards, rating badges, match % pills, trailer links, and storyline expanders.
   - TMDB API dynamic poster fetching with local caching and offline fallbacks.

5. **PEAS Architecture & NLP Sandbox**:
   - PEAS Agent model breakdown (Performance, Environment, Actuators, Sensors).
   - Interactive Porter Stemming sandbox to test morphological word reduction live.

---

## 🚀 Quick Start Guide

### 1. Launch with One Click (Windows)
Double-click `run_app.bat` to launch the application.

### 2. Manual Terminal Launch
```bash
streamlit run app.py
```

---

## 👥 Project Team
- Sanjay Gedela (23BKT0139)
- Ishan Singh (23BCI0104)
- Khushi Singh (21BCT0364)
- Shrutika (23BCE0116)
- Jain Utkarsh Sandeep (23BCT0088)
