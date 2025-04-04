import pandas as pd
import numpy as np
import random
from collections import Counter
from tqdm import tqdm

random.seed(42)

# Load dataset
df = pd.read_csv("./dataset/edited_ratings.csv")
print("Original ratings:", len(df))

# Only use positive interactions
positive_df = df[df["rating"] >= 4].copy()
positive_df["label"] = 1
print("Positive interactions:", len(positive_df))

# Create user -> set of interacted items
user_pos_dict = (
    positive_df.groupby("userId")["movie_idx"]
    .apply(set)
    .to_dict()
)

# Get item popularity (number of interactions)
item_popularity = Counter(df["movie_idx"])
popular_items = [item for item, _ in item_popularity.most_common(1000)]  # top 1000 popular items

# Generate negatives by sampling popular items not yet interacted with
negatives = []
for user, pos_items in tqdm(user_pos_dict.items(), desc="Generating negatives"):
    for pos_item in pos_items:
        negatives_for_user = []
        attempts = 0
        while len(negatives_for_user) < 4 and attempts < 20:
            sampled_item = random.choice(popular_items)
            if sampled_item not in pos_items:
                negatives_for_user.append(sampled_item)
            attempts += 1
        for neg_item in negatives_for_user:
            negatives.append([user, neg_item, 0])

neg_df = pd.DataFrame(negatives, columns=["userId", "movie_idx", "label"])
print("Generated smart negatives:", len(neg_df))

# Combine and shuffle
final_df = pd.concat([positive_df[["userId", "movie_idx", "label"]], neg_df], ignore_index=True)
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Add unique ID
final_df.insert(0, "id", range(len(final_df)))

# Save to CSV
final_df.to_csv("./dataset/with_negatives.csv", index=False)
print("generate negative samplings")
