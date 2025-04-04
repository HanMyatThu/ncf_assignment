import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from ncf import NCF

def evaluate(model, test_df, device, k=10):
    model.eval()
    all_items = set(test_df["movie_idx"].unique())  # All unique items in the test data
    user_items = test_df.groupby("userId")["movie_idx"].apply(set).to_dict()  # User to items map
    
    recalls, ndcgs = [], []
    
    for user in tqdm(user_items):  # Iterate over each user in the test data
        pos_items = user_items[user]  # Positive items for this user
        neg_items = list(all_items - pos_items)  # Negative items (all items except the user's positive items)
        
        # Make sure we have enough negative samples
        if len(neg_items) < 99:
            continue  # Skip if there are not enough negative samples
        
        test_items = list(pos_items) + np.random.choice(neg_items, 99, replace=False).tolist()  # 99 negative samples
        
        with torch.no_grad():
            user_tensor = torch.LongTensor([user] * len(test_items)).to(device) 
            item_tensor = torch.LongTensor(test_items).to(device) 
            scores = model(user_tensor, item_tensor).cpu().numpy()  
            
        ranked = np.argsort(-scores) 
        hits = [1 if test_items[i] in pos_items else 0 for i in ranked[:k]] 
        
        recall = sum(hits) / min(len(pos_items), k)  # Recall@k
        dcg = sum([hit / np.log2(i + 2) for i, hit in enumerate(hits)])  # DCG@k
        idcg = sum([1 / np.log2(i + 2) for i in range(min(len(pos_items), k))])  # Ideal DCG@k
        
        recalls.append(recall)
        ndcgs.append(dcg / (idcg + 1e-8)) 
    
    return np.mean(recalls), np.mean(ndcgs)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test data
    test_df = pd.read_csv("./dataset/test.csv")
    
    # Initialize model
    num_users = test_df["userId"].max() + 1  
    num_items = test_df["movie_idx"].max() + 1 
    model = NCF(num_users, num_items).to(device)  
    model.load_state_dict(torch.load("ncf_model.pt",weights_only=True)) 
    
    # Evaluate the model
    recall, ndcg = evaluate(model, test_df, device)
    print(f"Recall@{10}: {recall:.4f}, NDCG@{10}: {ndcg:.4f}")
