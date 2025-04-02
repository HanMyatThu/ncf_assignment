# for my machine (window)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
from torch.utils.data import DataLoader
from ncf import NCF 
from dataloader import NCFDataset  
from train_loop import train_model
import torch

# load datasets
train_df = pd.read_csv("./dataset/train.csv")
val_df = pd.read_csv("./dataset/val.csv")
test_df = pd.read_csv("./dataset/test.csv")

# get no of items, and no of users
num_users = max(
    train_df["userId"].max(),
    val_df["userId"].max(),
    test_df["userId"].max()
) + 1

num_items = max(
    train_df["movie_idx"].max(),
    val_df["movie_idx"].max(),
    test_df["movie_idx"].max()
) + 1


# load datasets with dataloader
train_dataset = NCFDataset(train_df)
val_dataset = NCFDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256)

# model initialization
model = NCF(num_users=num_users, num_items=num_items)

# train the model
device = "cuda" if torch.cuda.is_available() else "cpu"
print('before training!')
train_model(model, train_loader, val_loader, device=device)
