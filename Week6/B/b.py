

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np


# Load datasets
movies_df = pd.read_csv('movies.csv')
movies_df.head(5)

ratings_df = pd.read_csv('ratings.csv')
ratings_df.head(5)

# Subset for user 2
u2ratings = ratings_df[ratings_df['userId'] == 2]
u2ratings

# Task 1: Number of movies watched by user 2
nm = u2ratings.shape[0]
print("Number of movies watched by user2: ", nm)

# Task 2: Bar chart of user 2's movie ratings
plt.figure(figsize=(8, 6))
sns.countplot(x='rating', data=u2ratings, palette='viridis', hue='rating', legend=False)
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Movie Ratings for User 2')
plt.show()

# Distribution of movie genres
plt.figure(figsize=(10, 6))
gc = movies_df['genres'].str.split('|', expand=True).stack().value_counts()
gc.plot(kind='bar', color='skyblue')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.title('Distribution of Movie Genres')
plt.xticks(rotation=45)
plt.show()

# Task 3: Top movies of user 2
topmovies = u2ratings[u2ratings['rating'] >= 4.5].merge(movies_df, on='movieId')[['title', 'rating']].sort_values(by='rating', ascending=False)
print("User 2's top movies:")
print(topmovies)

# Task 4: Find the most similar user to user 2 using cosine and Euclidean distances
u2vector = u2ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
all_users_vector = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)

cmn_columns = u2vector.columns.intersection(all_users_vector.columns)

u2vector = u2vector[cmn_columns]
all_users_vector = all_users_vector[cmn_columns]

cos_similarity = cosine_similarity(u2vector, all_users_vector) 
euclidean_distances = euclidean_distances(u2vector, all_users_vector)

print(f"Most similar user to User 2 (cosine): {cos_similarity}")
print(f"Most similar user to User 2 (euclidean): {euclidean_distances}")

cos_similarity[:, 1] = -1 
euclidean_distances[:, 1] = np.inf 

ms_cos_index = np.argmax(cos_similarity)
ms_cos_score = cos_similarity[0, ms_cos_index]

print(f"Most similar user to User 2 (cosine): User {ms_cos_index + 1} with similarity score {ms_cos_score}")

ms_euclidean_index = np.argmin(euclidean_distances)
ms_euclidean_distance = euclidean_distances[0, ms_euclidean_index]

print(f"Most similar user to User 2 (euclidean): User {ms_euclidean_index + 1} with distance {ms_euclidean_distance}")

# Extract movies rated by the most similar user
most_similar_user_ratings = ratings_df[ratings_df['userId'] == (ms_cos_index + 1)]

# Merge with movies_df to include movie details
most_similar_user_ratings_with_details = most_similar_user_ratings.merge(movies_df, on='movieId')

print("Movies rated by the most similar user:")
print(most_similar_user_ratings_with_details)

# Task 5: Recommend movies for user 2 using cosine similarity
user_2_recommended_movies = all_users_vector.loc[ms_cos_index]
movies_not_rated_by_user_2 = user_2_recommended_movies[user_2_recommended_movies == 0].index

# threshold rating
threshold_rating = 3.0

# Extract recommended movies information with ratings above the threshold
recommended_movies_info = movies_df[movies_df['movieId'].isin(movies_not_rated_by_user_2)]
recommended_movies_info = recommended_movies_info[['movieId', 'title']].merge(ratings_df[ratings_df['userId'] == (ms_cos_index + 1)], on='movieId')
recommended_movies_info = recommended_movies_info.rename(columns={'rating': 'user_rating'})

# Filter movies above threshold
recommended_movies_info = recommended_movies_info[recommended_movies_info['user_rating'] >= threshold_rating]

print("Movies recommended for User 2 based on similarity with User", ms_cos_index + 1, "with ratings above", threshold_rating)
print(recommended_movies_info)



