# 1_data_exploration.ipynb

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Sample movies.csv content
movies_csv = '''movie_id,title,genres,keywords,cast,overview
1,The Shawshank Redemption,"Drama|Crime","prison|escape|hope","Tim Robbins|Morgan Freeman","Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency."
2,The Dark Knight,"Action|Crime|Drama","vigilante|joker|batman","Christian Bale|Heath Ledger","When the menace known as the Joker emerges, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice."
3,Inception,"Action|Sci-Fi|Thriller","dream|subconscious|heist","Leonardo DiCaprio|Joseph Gordon-Levitt","A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a CEO."
4,Forrest Gump,"Drama|Romance","love|war|life","Tom Hanks|Robin Wright","The presidencies of Kennedy and Johnson, the Vietnam War, the Watergate scandal and other historical events unfold through the perspective of an Alabama man with an IQ of 75."
5,The Matrix,"Action|Sci-Fi|Adventure","simulation|ai|reality","Keanu Reeves|Laurence Fishburne","A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers."
6,Gladiator,"Action|Drama|Adventure","gladiator|roman empire|revenge","Russell Crowe|Joaquin Phoenix","A former Roman General sets out to exact vengeance against the corrupt emperor who murdered his family and sent him into slavery."
7,Titanic,"Drama|Romance","shipwreck|love|tragedy","Leonardo DiCaprio|Kate Winslet","A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic."
8,The Avengers,"Action|Adventure|Sci-Fi","superhero|team|marvel","Robert Downey Jr.|Chris Evans","Earth's mightiest heroes must come together and learn to fight as a team to stop the mischievous Loki and his alien army from enslaving humanity."
9,Interstellar,"Adventure|Drama|Sci-Fi","space|black hole|time dilation","Matthew McConaughey|Anne Hathaway","A team of explorers undertakes a mission through a wormhole in space in an attempt to ensure humanity’s survival."
10,Jurassic Park,"Adventure|Sci-Fi|Thriller","dinosaurs|island|escape","Sam Neill|Laura Dern","During a preview tour, a theme park suffers a major power breakdown that allows its cloned dinosaur exhibits to run amok."
11,The Lion King,"Animation|Adventure|Drama","africa|lion|kingdom","Matthew Broderick|Jeremy Irons","Lion prince Simba and his father are targeted by his bitter uncle, who wants to ascend the throne himself."
12,Fight Club,"Drama","insomnia|rebellion|psychological","Edward Norton|Brad Pitt","An insomniac office worker and a devil-may-care soap maker form an underground fight club that evolves into something much more."
13,Avengers: Endgame,"Action|Adventure|Drama","superhero|time travel|marvel","Robert Downey Jr.|Chris Evans","After the devastating events of Avengers: Infinity War, the universe is in ruins. The Avengers assemble once more in order to reverse Thanos' actions."
14,The Social Network,"Biography|Drama","facebook|founder|lawsuit","Jesse Eisenberg|Andrew Garfield","Harvard student Mark Zuckerberg creates the social networking site that would become known as Facebook, and is later sued by two brothers who claimed he stole their idea."
15,La La Land,"Comedy|Drama|Musical","jazz|love|dream","Ryan Gosling|Emma Stone","While navigating their careers in Los Angeles, a musician and an aspiring actress fall in love while attempting to reconcile their aspirations for the future."
'''

# Sample keywords.csv content
keywords_csv = '''movie_id,keyword
1,prison
1,escape
1,hope
2,vigilante
2,joker
2,batman
3,dream
3,subconscious
3,heist
4,love
4,war
4,life
5,simulation
5,ai
5,reality
6,gladiator
6,roman empire
6,revenge
7,shipwreck
7,love
7,tragedy
8,superhero
8,team
8,marvel
9,space
9,black hole
9,time dilation
10,dinosaurs
10,island
10,escape
11,africa
11,lion
11,kingdom
12,insomnia
12,rebellion
12,psychological
13,superhero
13,time travel
13,marvel
14,facebook
14,founder
14,lawsuit
15,jazz
15,love
15,dream
'''

# Sample ratings.csv content
ratings_csv = '''user_id,movie_id,rating
1,1,5
1,2,4
1,3,5
2,2,5
2,4,4
2,5,3
3,1,4
3,5,5
3,6,4
4,7,5
4,8,4
4,9,3
5,10,4
5,11,5
5,12,3
'''

# Load datasets from strings
movies = pd.read_csv(StringIO(movies_csv))
keywords = pd.read_csv(StringIO(keywords_csv))
ratings = pd.read_csv(StringIO(ratings_csv))

# Display basic info and preview
print("Movies Dataset:")
print(movies.info())
print(movies.head())

print("\nKeywords Dataset:")
print(keywords.info())
print(keywords.head())

print("\nRatings Dataset:")
print(ratings.info())
print(ratings.head())

# Check for missing values
print("\nMissing Values per Dataset:")
print('Movies:\n', movies.isnull().sum())
print('Keywords:\n', keywords.isnull().sum())
print('Ratings:\n', ratings.isnull().sum())

# Summary statistics for ratings
print("\nRatings Summary Statistics:")
print(ratings['rating'].describe())

# Distribution of movie genres (split genres to count)
movies['genres_list'] = movies['genres'].apply(lambda x: x.split('|'))
all_genres = movies['genres_list'].explode()
plt.figure(figsize=(12,6))
sns.countplot(y=all_genres, order=all_genres.value_counts().index)
plt.title('Genre Distribution')
plt.xlabel('Count')
plt.ylabel('Genres')
plt.show()

# Distribution of keywords
plt.figure(figsize=(12,6))
top_keywords = keywords['keyword'].value_counts().head(20)
sns.barplot(y=top_keywords.index, x=top_keywords.values)
plt.title('Top 20 Keywords')
plt.xlabel('Count')
plt.ylabel('Keyword')
plt.show()

# Ratings distribution
plt.figure(figsize=(8,4))
sns.countplot(x='rating', data=ratings, palette='viridis')
plt.title('Ratings Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# Number of ratings per user
ratings_per_user = ratings.groupby('user_id')['rating'].count()
plt.figure(figsize=(8,4))
sns.histplot(ratings_per_user, bins=20, kde=False)
plt.title('Number of Ratings per User')
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Users')
plt.show()

# Number of ratings per movie
ratings_per_movie = ratings.groupby('movie_id')['rating'].count()
plt.figure(figsize=(8,4))
sns.histplot(ratings_per_movie, bins=20, kde=False)
plt.title('Number of Ratings per Movie')
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Movies')
plt.show()

# Remove temporary 'genres_list' helper column
movies.drop(columns=['genres_list'], inplace=True)
