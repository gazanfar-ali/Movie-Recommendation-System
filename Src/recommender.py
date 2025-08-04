# src/recommender.py

from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class ContentBasedRecommender:
    def __init__(self, movies_df, feature_matrix):
        self.movies = movies_df.reset_index(drop=True)
        self.feature_matrix = feature_matrix
        self.cosine_sim = cosine_similarity(self.feature_matrix, self.feature_matrix)
        self.indices = pd.Series(self.movies.index, index=self.movies['title']).drop_duplicates()
    
    def get_recommendations(self, title, top_n=5):
        idx = self.indices.get(title)
        if idx is None:
            return f"Movie '{title}' not found in database."
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        movie_indices = [i[0] for i in sim_scores]
        return self.movies['title'].iloc[movie_indices].tolist()
