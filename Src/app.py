# src/app.py

import streamlit as st
import pandas as pd
from data_preprocessing import load_data, clean_data, merge_features
from feature_engineering import vectorize_features
from recommender import ContentBasedRecommender

@st.cache_data
def load_and_prepare():
    movies, keywords, ratings = load_data('data/movies.csv', 'data/keywords.csv', 'data/ratings.csv')
    movies = clean_data(movies)
    movies = merge_features(movies)
    tfidf_matrix, tfidf = vectorize_features(movies)
    recommender = ContentBasedRecommender(movies, tfidf_matrix)
    return recommender

def main():
    st.title("Movie Recommendation System (Content-Based Filtering)")
    recommender = load_and_prepare()
    
    movie_list = recommender.movies['title'].tolist()
    selected_movie = st.selectbox("Select a movie you like:", movie_list)
    
    if st.button('Show Recommendations'):
        recommendations = recommender.get_recommendations(selected_movie, top_n=5)
        if isinstance(recommendations, str):
            st.warning(recommendations)
        else:
            st.write("Recommended Movies:")
            for i, movie in enumerate(recommendations, 1):
                st.write(f"{i}. {movie}")

if __name__ == '__main__':
    main()
