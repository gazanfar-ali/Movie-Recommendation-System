# 🎬 Movie Recommendation System (Content-Based Filtering)

Welcome to your own personalized movie recommender! This project guides you through building a content-based movie recommendation engine using **Python**, with a focus on clean data pipelines, modular code, and user-friendly outputs. Inspired by the algorithms behind Netflix and IMDB, this repository is designed for learning, showcasing, and even expanding into your own app!

---

## ✨ Project Highlights

- **Intuitive Recommendations:** Get smart movie suggestions based on genres, keywords, and cast similarity.
- **Complete ML Pipeline:** From raw CSVs to a deployable app, all steps are modular and transparent.
- **Interactive Demo:** Try the ready-made Streamlit app to see instant recommendations.
- **Beginner-Friendly:** Well-documented notebooks, clear explanations, and starter datasets.
- **Easily Extensible:** Add more movies, try new features, or even blend in collaborative filtering.

---

## 🚦 Motivation

Ever wanted Netflix-style “Because you watched…” recommendations, but under your control and fully explainable? This project:
- Shows how real-world movie recommenders work beneath the hood,
- Teaches you the full workflow: data wrangling → feature engineering → ML modeling → app,
- Builds a portfolio-ready project suitable for internships, job showcases, and learning advanced Python.

---

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas** and **NumPy** for fast and flexible data handling
- **Scikit-learn** for feature engineering (TF-IDF, CountVectorizer) and cosine similarity
- **Matplotlib**/**Seaborn** for stunning data visualizations
- **Streamlit** for your interactive web app

---

## 📊 Project Structure

```bash
movie-recommendation-system/
├── data/ # Raw and processed datasets (movies.csv, keywords.csv, ratings.csv, etc.)
├── notebooks/ # Exploration, cleaning, feature engineering, modeling (stepwise .ipynb files)
├── src/ # Reusable code modules:
│ ├── data_preprocessing.py
│ ├── feature_engineering.py
│ ├── recommender.py
│ └── app.py
├── requirements.txt # All dependencies for environment setup
├── LICENSE # MIT License
├── README.md # Full documentation (this file)
```

---

## 📥 Quickstart

1. **Clone the Repository:**
```bash
git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system
```

2. **Install Requirements:**
``` bash
pip install -r requirements.txt
```


3. **Explore the Notebooks:**
- Each notebook in `/notebooks` covers a major part of the pipeline: EDA, preprocessing, feature engineering, or modeling.

4. **Try the Live Demo (Streamlit):**
```bash
streamlit run src/app.py
```

- Select your favorite movie and see the top content-based recommendations!

---

## 👀 Visual Workflow
```bash
Raw Data (CSV)
│
▼
[Exploration] → [Cleaning] → [Feature Merging]
▼
[Text Vectorization: TF-IDF]
▼
[Cosine Similarity Calculation]
▼
[Get Recommendations!]
```


---

## 🧑‍💻 Example Usage
```bash
from src.data_preprocessing import load_data, clean_data, merge_features
from src.feature_engineering import vectorize_features
from src.recommender import ContentBasedRecommender

Load & preprocess data
movies, _, _ = load_data('data/movies.csv')
movies = clean_data(movies)
movies = merge_features(movies)

Feature engineering
tfidf_matrix, _ = vectorize_features(movies)

Build recommender
recommender = ContentBasedRecommender(movies, tfidf_matrix)
print(recommender.get_recommendations('The Matrix'))

```

---

## 🔎 FAQs

- **Q: Can I add more movies?**  
  Yes! Just expand `movies.csv` and rerun the notebooks or app.

- **Q: I want collaborative filtering!**  
  The code is designed for modularity. Try extending with the included `ratings.csv` or add a new pipeline.

- **Q: Will this scale to 1000s of movies?**  
  For small datasets, yes! For large-scale, see notes in each notebook about performance and engineering.

---

## 📄 License

MIT — see the [LICENSE](./LICENSE) file for details.

---

**Enjoy building, experimenting, and learning! If you like this repository, star it and share your own recommender projects.**


