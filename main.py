import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

data = {
    'user': ['Ana', 'Ana', 'Matheus', 'Matheus', 'Nalu', 'Nalu'],
    'movie': ['Final Destination 6', 'The Conjuring', 'Final Destination 6', 'Lorax', 'The Conjuring', 'Lorax'],
    'rating': [5, 3, 4, 2, 4, 5]
}

df = pd.DataFrame(data)

user_item_matrix = df.pivot_table(index='user', columns='movie', values='rating', fill_value=0)

item_similarity = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

def recommend_similar_items(selected_movie, similarity_df, top_n=3):
    if selected_movie not in similarity_df.index:
        return pd.Series(dtype=float)
    similar_scores = similarity_df[selected_movie].sort_values(ascending=False)
    similar_scores = similar_scores.drop(selected_movie)
    return similar_scores.head(top_n)

st.set_page_config(page_title="🎥 Item-Based Movie Recommender", layout="centered")
st.title("🎬 Movie Recommender (Item-Based Filtering)")
st.markdown("Recommend movies based on item similarity (cosine similarity of ratings).")

selected_movie = st.selectbox("Select a movie to find similar ones:", user_item_matrix.columns)

top_n = st.slider("How many recommendations?", min_value=1, max_value=5, value=3)

if st.button("Get Recommendations"):
    recommendations = recommend_similar_items(selected_movie, item_similarity_df, top_n=top_n)
    if recommendations.empty:
        st.warning("No similar movies found.")
    else:
        st.subheader("Similar Movies:")
        for movie, score in recommendations.items():
            st.markdown(f"**{movie}** — Similarity: {score:.2f}")

with st.expander("📊 Show raw data and matrices"):
    st.subheader("Original Data:")
    st.dataframe(df)

    st.subheader("User-Item Matrix:")
    st.dataframe(user_item_matrix)

    st.subheader("Item Similarity Matrix:")
    st.dataframe(item_similarity_df.round(2))
