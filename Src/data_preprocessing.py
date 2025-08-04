# src/data_preprocessing.py

import pandas as pd

def load_data(movies_path, keywords_path=None, ratings_path=None):
    movies = pd.read_csv(movies_path)
    keywords = pd.read_csv(keywords_path) if keywords_path else None
    ratings = pd.read_csv(ratings_path) if ratings_path else None
    return movies, keywords, ratings

def clean_data(movies):
    # Fill missing text columns with empty strings
    text_columns = ['genres', 'keywords', 'cast', 'overview']
    for col in text_columns:
        if col in movies.columns:
            movies[col] = movies[col].fillna('')
    return movies

def merge_features(movies):
    # Combine genres, keywords, and cast into a single string feature
    def combine(row):
        return row['genres'] + ' ' + row['keywords'].replace('|', ' ') + ' ' + row['cast'].replace('|', ' ')
    movies['combined_features'] = movies.apply(combine, axis=1)
    return movies
