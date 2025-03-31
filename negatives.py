import pandas as pd
import numpy as np
from tqdm import tqdm

df = pd.read_csv('./dataset/edited_ratings.csv')

user_positive_items = df.groupby('userId')['movie_idx'].apply(set).to_dict()
all_items = set(df['movie_idx'].unique())

negatives = []
num_neg_per_pos = 4 

print("Generating negative samples...")
for user in tqdm(user_positive_items):
    pos_items = user_positive_items[user]
    num_pos = len(pos_items)
    neg_candidates = list(all_items - pos_items)
    
    # randomly select items
    neg_items = np.random.choice(neg_candidates, size=num_pos * num_neg_per_pos, replace=True)
    
    for item in neg_items:
        negatives.append([user, item, 0]) 

df['label'] = 1 
df_neg = pd.DataFrame(negatives, columns=['userId', 'movie_idx', 'label'])

df_combined = pd.concat([df[['userId', 'movie_idx', 'label']], df_neg], ignore_index=True)
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

# add index in the records
df_combined.insert(0, 'id', range(len(df_combined))) 

df_combined.to_csv('./dataset/with_negatives.csv', index=False)
