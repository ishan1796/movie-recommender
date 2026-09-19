import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

def generate_pca_comparison_plot(df, vectors_data, movie_titles, title_label="Content-Based Comparison"):
    """
    Reproduces and enhances the PCA 2D Similarity graphs from PPT Slides 7 & 8.
    Takes a list of movie titles, extracts their high-dimensional TF-IDF/Count vectors,
    projects down to 2 principal components (PCA Component 1 vs PCA Component 2),
    and creates an interactive Plotly scatter plot.
    """
    title_to_idx = {t.strip().lower(): idx for idx, t in enumerate(df["title"])}
    
    selected_indices = []
    found_titles = []
    
    for t in movie_titles:
        clean = t.strip().lower()
        if clean in title_to_idx:
            selected_indices.append(title_to_idx[clean])
            found_titles.append(df.iloc[title_to_idx[clean]]["title"])
        else:
            # Fuzzy match
            for stored_clean, idx in title_to_idx.items():
                if clean in stored_clean:
                    if idx not in selected_indices:
                        selected_indices.append(idx)
                        found_titles.append(df.iloc[idx]["title"])
                    break

    if len(selected_indices) < 2:
        return None, "Please select at least 2 matching movies for 2D PCA projection."

    # Extract vectors
    if "tfidf_vectors" in vectors_data:
        feature_matrix = vectors_data["tfidf_vectors"][selected_indices]
    elif "count_vectors" in vectors_data:
        feature_matrix = vectors_data["count_vectors"][selected_indices]
    else:
        return None, "Vector data not available."

    # Compute PCA to 2 components
    n_components = min(2, len(selected_indices), feature_matrix.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    pca_result = pca.fit_transform(feature_matrix)

    if n_components == 1:
        pca_x = pca_result[:, 0]
        pca_y = np.zeros(len(selected_indices))
    else:
        pca_x = pca_result[:, 0]
        pca_y = pca_result[:, 1]

    sub_df = df.iloc[selected_indices].copy()
    sub_df["PCA Component 1"] = pca_x
    sub_df["PCA Component 2"] = pca_y
    sub_df["Genres"] = sub_df["genres_list"].apply(lambda g: ", ".join(g) if isinstance(g, list) else "")
    sub_df["Director"] = sub_df["director"]
    sub_df["Rating"] = sub_df["vote_average"]
    sub_df["Year"] = sub_df["release_year"]

    # Build Interactive Plotly Figure styled like Netflix / Modern Dark Theme
    fig = go.Figure()

    colors = px.colors.qualitative.Bold + px.colors.qualitative.Vivid

    for i, row in enumerate(sub_df.itertuples()):
        c = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=[row._11 if hasattr(row, '_11') else row.Index], # dynamically mapped below
            y=[0],
            mode="markers+text",
            name=row.title,
            text=[f"<b>{row.title}</b>"],
            textposition="top center",
            marker=dict(
                size=18,
                color=c,
                line=dict(width=2, color="#ffffff"),
                opacity=0.9
            ),
            hovertemplate=(
                f"<b>{row.title}</b> ({row.release_year})<br>"
                f"Rating: ⭐ {row.vote_average}/10<br>"
                f"Genres: {', '.join(row.genres_list) if isinstance(row.genres_list, list) else ''}<br>"
                f"Director: {row.director}<br>"
                f"PCA 1: %{{x:.2f}}<br>PCA 2: %{{y:.2f}}<extra></extra>"
            )
        ))

    # Re-plot accurately with exact coordinates
    fig.data = []
    for i, (idx, row) in enumerate(sub_df.iterrows()):
        c = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=[row["PCA Component 1"]],
            y=[row["PCA Component 2"]],
            mode="markers+text",
            name=row["title"],
            text=[f"  {row['title']}"],
            textposition="middle right",
            textfont=dict(size=12, color="#ffffff"),
            marker=dict(
                size=18,
                color=c,
                line=dict(width=2, color="#ffffff"),
                opacity=0.95
            ),
            hovertemplate=(
                f"<b>{row['title']}</b> ({row['release_year']})<br>"
                f"⭐ Rating: {row['vote_average']}/10<br>"
                f"🎬 Director: {row['director']}<br>"
                f"🎭 Genres: {', '.join(row['genres_list']) if isinstance(row['genres_list'], list) else ''}<extra></extra>"
            )
        ))

    fig.update_layout(
        title=dict(
            text=f"<b>{title_label} (2D PCA Semantic Space)</b>",
            font=dict(size=18, color="#ffffff"),
            x=0.5
        ),
        xaxis_title=dict(text="PCA Component 1", font=dict(size=14, color="#e0e0e0")),
        yaxis_title=dict(text="PCA Component 2", font=dict(size=14, color="#e0e0e0")),
        paper_bgcolor="#141414",
        plot_bgcolor="#1f1f1f",
        xaxis=dict(
            gridcolor="#333333",
            zerolinecolor="#666666",
            tickfont=dict(color="#b0b0b0")
        ),
        yaxis=dict(
            gridcolor="#333333",
            zerolinecolor="#666666",
            tickfont=dict(color="#b0b0b0")
        ),
        showlegend=True,
        legend=dict(
            font=dict(color="#ffffff"),
            bgcolor="rgba(20,20,20,0.8)",
            bordercolor="#444444",
            borderwidth=1
        ),
        height=550,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig, None
