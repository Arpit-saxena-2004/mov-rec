import streamlit as st
import pickle
import numpy as np



from sentence_transformers import SentenceTransformer
import os

# Force sentence-transformers to use CPU, even if torch sees a 'cuda'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model = SentenceTransformer('all-MiniLM-L6-v2')



# Load similarity matrix and movies data with pickle
similarity = pickle.load(open("similarity_matrix.pkl", "rb"))


with open("movies_data.pkl", "rb") as f:
    new_df = pickle.load(f)


# Define recommendation function
def recommend(movie):
    movie = movie.lower()
    if movie not in new_df["original_title"].str.lower().values:
        return [("Movie not found in dataset", 0.0)]
    
    index = new_df[new_df["original_title"].str.lower() == movie].index[0]
    distances = list(enumerate(similarity[index]))
    movies = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
    
    return [(new_df.iloc[i[0]]['original_title'], i[1] * 100) for i in movies]

# Streamlit UI
st.title("🎬 Movie Recommender System")
movie_name = st.text_input("Enter a movie name:")

if st.button("Get Recommendations"):
    if movie_name:
        recommendations = recommend(movie_name)
        st.write("**Recommended Movies with Similarity Scores:**")
        for title, score in recommendations:
            st.write(f"🎥 {title} — & Your chances of liking it are **{score:.0f}%**")

    else:
        st.write("oops soory movie is not present in the dataset!")


st.write("REQUEST : I kindly request you to search for movies other than those in Hindi, as the recommender system is not efficient with Hindi-language movies due to limited and insufficient data. Thank you for your cooperation!")