import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

# Load datasets
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Subset for user 2
user_2_ratings = ratings[ratings['userId'] == 2]

# Task 1: How many movies has user 2 watched?
num_movies_watched = len(user_2_ratings)
print(f"User 2 has watched {num_movies_watched} movies.")

# Task 2: Plot a bar chart of user 2's movie ratings
rating_counts = user_2_ratings['rating'].value_counts().sort_index()
rating_counts.plot(kind='bar', color='skyblue')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Movie Ratings for User 2')
plt.show()

# Task 3: What are some of user 2's top movies?
user_2_top_movies = user_2_ratings.merge(movies, on='movieId')[['title', 'rating']].sort_values(by='rating', ascending=False)
print("User 2's top movies:")
print(user_2_top_movies.head())

# Task 4: Find the most similar user to user 2 using cosine and manhattan distances
user_2_vector = user_2_ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
all_users_vector = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

common_columns = user_2_vector.columns.intersection(all_users_vector.columns)

user_2_vector = user_2_vector[common_columns]
all_users_vector = all_users_vector[common_columns]

cosine_similarities = cosine_similarity(user_2_vector, all_users_vector)
manhattan_distances = pairwise_distances(user_2_vector, all_users_vector, metric='manhattan')

print(f"Most similar user to User 2 (cosine): {cosine_similarities}")
print(f"Most similar user to User 2 (manhattan): {manhattan_distances}")

# Task 5: Recommend movies for user 2 using cosine similarity
import numpy as np

# Exclude User 2 from the calculation
cosine_similarities[:, 1] = -1 
manhattan_distances[:, 1] = np.inf 

# index of the most similar user using cosine similarity
most_similar_cosine_index = np.argmax(cosine_similarities)
most_similar_cosine_score = cosine_similarities[0, most_similar_cosine_index]

print(f"Most similar user to User 2 (cosine): User {most_similar_cosine_index + 1} with similarity score {most_similar_cosine_score}")

# index of the most similar user using Manhattan distance
most_similar_manhattan_index = np.argmin(manhattan_distances)
most_similar_manhattan_distance = manhattan_distances[0, most_similar_manhattan_index]

print(f"Most similar user to User 2 (manhattan): User {most_similar_manhattan_index + 1} with distance {most_similar_manhattan_distance}")

user_2_recommended_movies = all_users_vector.loc[most_similar_cosine_index]

movies_not_rated_by_user_2 = user_2_recommended_movies[user_2_recommended_movies == 0].index

# Extract recommended movies information
recommended_movies_info = movies[movies['movieId'].isin(movies_not_rated_by_user_2)]
recommended_movies_info = recommended_movies_info[['movieId', 'title']].merge(ratings[ratings['userId'] == most_similar_cosine_index + 1], on='movieId')
recommended_movies_info = recommended_movies_info.rename(columns={'rating': 'user_rating'})

print("Movies rated by the most similar user:")
print(recommended_movies_info)

recommended_movies_info = movies[movies['movieId'].isin(user_2_recommended_movies[user_2_recommended_movies == 0].index)]

print("Movies not yet rated by User 2:")
print(recommended_movies_info)

# Get movies rated by the most similar user but not yet rated by User 2
movies_not_rated_by_user_2 = user_2_recommended_movies[user_2_recommended_movies == 0].index

# Extract recommended movies information
recommended_movies_info = movies[movies['movieId'].isin(movies_not_rated_by_user_2)]

# Merge with ratings of the most similar user
recommended_movies_info = recommended_movies_info.merge(
    ratings[ratings['userId'] == most_similar_cosine_index + 1],
    on='movieId',
    how='left'
)

# Filter only movies that are highly rated by the most similar user
threshold_rating = 3.5
recommended_movies_info = recommended_movies_info[recommended_movies_info['rating'] >= threshold_rating]

print("Recommended movies for User 2:")
print(recommended_movies_info[['movieId', 'title', 'genres', 'rating']])

