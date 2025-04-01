# for my machine (window)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from ncf import NCF  

# dataset loading
train_df = pd.read_csv("./dataset/train.csv")
val_df = pd.read_csv("./dataset/val.csv")
test_df = pd.read_csv("./dataset/test.csv")

num_users = max(train_df["userId"].max(), val_df["userId"].max(), test_df["userId"].max()) + 1
num_items = max(train_df["movie_idx"].max(), val_df["movie_idx"].max(), test_df["movie_idx"].max()) + 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model loading
model = NCF(num_users, num_items, embedding_dim=64, mlp_dims=[128, 64, 32], dropout=0.2)
model.load_state_dict(torch.load("ncf_model.pt", map_location=device, weights_only=True))
model.to(device)
model.eval()
print("✅ Model loaded.")

# build user item history
all_items = set(range(num_items))
interacted = (
    pd.concat([train_df, val_df, test_df])
    .groupby("userId")["movie_idx"]
    .apply(set)
    .to_dict()
)

# 100 users
user_test_items = test_df.groupby("userId")["movie_idx"].apply(list).to_dict()
num_negatives = 99  

# recall@10
def recall_at_k(ranked_list, ground_truth, k=10):
    return int(ground_truth in ranked_list[:k])

# ndcg@10 
def ndcg_at_k(ranked_list, ground_truth, k=10):
    if ground_truth in ranked_list[:k]:
        index = ranked_list.index(ground_truth)
        return 1 / np.log2(index + 2)  # index 0-based, so +2
    return 0

# initialize placeholder arrays
recalls = []
ndcgs = []

for user, positives in tqdm(user_test_items.items(), desc="Evaluating"):
    for pos_item in positives:
        negative_items = list(all_items - interacted[user])
        if len(negative_items) < num_negatives:
             # skip this user if he has few negatives
            continue 
        sampled_negatives = random.sample(negative_items, num_negatives)
        candidates = sampled_negatives + [pos_item]
        random.shuffle(candidates)

        user_tensor = torch.LongTensor([user] * len(candidates)).to(device)
        item_tensor = torch.LongTensor(candidates).to(device)

        with torch.no_grad():
            scores = model(user_tensor, item_tensor)

        top_indices = torch.topk(scores, k=10).indices.cpu().numpy()
        top_items = [candidates[i] for i in top_indices]

        recalls.append(recall_at_k(top_items, pos_item, k=10))
        ndcgs.append(ndcg_at_k(top_items, pos_item, k=10))

# console print result
print("Evaluation complete.")
print(f"Recall@10: {np.mean(recalls):.4f}")
print(f"NDCG@10: {np.mean(ndcgs):.4f}")
