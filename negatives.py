import pandas as pd
import numpy as np
from tqdm import tqdm

df = pd.read_csv("./dataset/editing_ratings.csv")
all_items = set(df["movie_idx"].unique())
user_items = df.groupby("userId")["movie_idx"].apply(set).to_dict()

negatives = []
num_neg_per_pos = 4 

for user in tqdm(user_items):
    pos_items = user_items[user]
    neg_candidates = list(all_items - pos_items)
    neg_samples = np.random.choice(neg_candidates, size=len(pos_items) * num_neg_per_pos, replace=True)
    
    for item in neg_samples:
        negatives.append([user, item, 0])

# Combine positives and negatives
df_neg = pd.DataFrame(negatives, columns=["userId", "movie_idx", "label"])
full_df = pd.concat([df[["userId", "movie_idx", "label"]], df_neg])

# Shuffle and save
full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
full_df.to_csv("./dataset/with_negatives.csv", index=False)
print(f"Generated {len(negatives)} negative samples")
