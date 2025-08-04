# src/feature_engineering.py

from sklearn.feature_extraction.text import TfidfVectorizer

def create_combined_features(df):
    # Assuming combined_features column already exists or create here if needed
    return df

def vectorize_features(df, feature_col='combined_features'):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df[feature_col])
    return tfidf_matrix, tfidf
