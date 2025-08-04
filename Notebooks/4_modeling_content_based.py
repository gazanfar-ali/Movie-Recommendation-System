# 4_modeling_content_based.ipynb

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample data similar to previous notebook
data = {
    'movie_id': [1, 2, 3, 4, 5],
    'title': ['The Shawshank Redemption', 'The Dark Knight', 'Inception', 'Forrest Gump', 'The Matrix'],
    'combined_features': [
        'Drama Crime prison escape hope Tim Robbins Morgan Freeman',
        'Action Crime Drama vigilante joker batman Christian Bale Heath Ledger',
        'Action Sci-Fi Thriller dream subconscious heist Leonardo DiCaprio Joseph Gordon-Levitt',
        'Drama Romance love war life Tom Hanks Robin Wright',
        'Action Sci-Fi Adventure simulation ai reality Keanu Reeves Laurence Fishburne'
    ]
}
movies = pd.DataFrame(data)

# Vectorize combined_features
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a Series mapping movie titles to indices
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim, movies=movies, indices=indices, top_n=5):
    # Get the index of the movie that matches the title
    idx = indices[title]
    
    # Get pairwise similarity scores for that movie with all others
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort movies based on similarity scores (descending)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Skip the first movie (itself) and get top_n recommendations
    sim_scores = sim_scores[1:top_n+1]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the top n most similar movie titles
    return movies['title'].iloc[movie_indices]

# Example usage
print("Recommendations for 'The Dark Knight':")
print(get_recommendations('The Dark Knight'))

print("\nRecommendations for 'Forrest Gump':")
print(get_recommendations('Forrest Gump'))
