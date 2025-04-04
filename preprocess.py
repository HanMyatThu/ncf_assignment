import pandas as pd

df = pd.read_csv("./dataset/ratings.csv")

# Convert to 0-based indexing
df["userId"] = df["userId"] - 1
unique_movies = df["movieId"].unique()
movie2idx = {movie: idx for idx, movie in enumerate(unique_movies)}
df["movie_idx"] = df["movieId"].map(movie2idx)

# Keep only positive interactions (ratings >=4)
df = df[df["rating"] >= 4].copy()
df["label"] = 1

# Save processed data
df.to_csv("./dataset/editing_ratings.csv", index=False)
print(f"Processed {len(df)} positive interactions")
