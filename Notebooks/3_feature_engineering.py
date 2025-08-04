# 3_feature_engineering.ipynb

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load processed data (you can load from CSV if saved, or use the previous combined_features DataFrame)
# Here, for demonstration, let's assume loading processed_movies.csv:
# movies = pd.read_csv('data/processed_movies.csv')

# For standalone running, we recreate sample minimal data (replace this with loading your processed CSV)
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

# Initialize TF-IDF Vectorizer to convert text to numerical features
tfidf = TfidfVectorizer(stop_words='english')

# Fit and transform combined features into TF-IDF feature matrix
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])

print("TF-IDF feature matrix shape:", tfidf_matrix.shape)

# Show feature names (tokens)
feature_names = tfidf.get_feature_names_out()
print("\nSample of features (tokens):")
print(feature_names[:20])

# Optionally, save tfidf_matrix or feature names for modeling (not shown here)

