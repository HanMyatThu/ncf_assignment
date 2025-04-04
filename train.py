# for my machine (window)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import pandas as pd
from torch.utils.data import DataLoader
from dataloader import NCFDataset
from ncf import NCF
from train_loop import train_model 

# Device setup (use CUDA if available, otherwise fallback to CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load datasets
train_df = pd.read_csv("./dataset/train.csv")
val_df = pd.read_csv("./dataset/val.csv")

# Prepare datasets and dataloaders
train_set = NCFDataset(train_df)
val_set = NCFDataset(val_df)
train_loader = DataLoader(train_set, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1024)

# Model initialization
num_users = train_df["userId"].max() + 1
num_items = train_df["movie_idx"].max() + 1
model = NCF(num_users, num_items).to(device)

# Optimizer and learning rate setup (included in the train_model function)
lr = 0.0005
patience = 5
num_epochs = 30

# Call train_model from train_loop.py
train_model(model, train_loader, val_loader, num_epochs=num_epochs, lr=lr, patience=patience, device=device)
