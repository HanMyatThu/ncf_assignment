import pandas as pd

df = pd.read_csv("./dataset/ratings.csv")

df.userId = df.userId - 1

unique_movie_ids = set(df.movieId.values)
movie2idx = {movie_id: idx for idx, movie_id in enumerate(unique_movie_ids)}
df['movie_idx'] = df.movieId.map(movie2idx)

# drop timestamp because i don't want to use
df = df.drop(columns=['timestamp'])

# ratings >= 4 as positive interactions (label = 1)
df = df[df['rating'] >= 4].copy()
df['label'] = 1

df.to_csv('./dataset/edited_ratings.csv', index=False)
print("Complete Pre Data Processing")