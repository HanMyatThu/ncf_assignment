import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ============= Data Loading and Preprocessing Functions =============

def convert_dat_to_csv():
    """Convert original .dat files to CSV format"""
    print("Converting DAT files to CSV...")
    
    # ratings
    ratings = pd.read_csv('./dataset/ratings.dat', sep='::', engine='python',
                          names=['userId', 'movieId', 'rating', 'timestamp'])
    ratings.to_csv('./dataset/ratings.csv', index=False)

    # users
    users = pd.read_csv('./dataset/users.dat', sep='::', engine='python',
                        names=['userId', 'gender', 'age', 'occupation', 'zipCode'])
    users.to_csv('./dataset/users.csv', index=False)

    # movies
    movies = pd.read_csv('./dataset/movies.dat', sep='::', engine='python',
                         names=['movieId', 'title', 'genres'], encoding='ISO-8859-1')
    movies.to_csv('./dataset/movies.csv', index=False)

    print("All DAT files are converted to CSV")


def preprocess_ratings():
    """Preprocess ratings data: convert to 0-based indexing and keep positive interactions"""
    print("Preprocessing ratings data...")
    
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
    
    return df


def generate_negatives(df):
    """Generate negative samples to balance the dataset"""
    print("Generating negative samples...")
    
    all_items = set(df["movie_idx"].unique())
    user_items = df.groupby("userId")["movie_idx"].apply(set).to_dict()

    negatives = []
    num_neg_per_pos = 8 

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
    
    return full_df


def split_dataset(df):
    """Split the dataset into train, validation and test sets"""
    print("Splitting dataset...")
    
    # Train (70%), Temp (30%)
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df["label"], random_state=42
    )

    # Val (15%), Test (15%)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
    )
    
    # Save splits
    train_df.to_csv("./dataset/train.csv", index=False)
    val_df.to_csv("./dataset/val.csv", index=False)
    test_df.to_csv("./dataset/test.csv", index=False)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df


# ============= Dataset Class =============

class NCFDataset(Dataset):
    """PyTorch Dataset for Neural Collaborative Filtering"""
    def __init__(self, df):
        self.users = torch.tensor(df["userId"].values, dtype=torch.long)
        self.items = torch.tensor(df["movie_idx"].values, dtype=torch.long)
        self.labels = torch.tensor(df["label"].values, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]


# ============= Model Definition =============

class NCF(nn.Module):
    """Neural Collaborative Filtering model combining GMF and MLP"""
    def __init__(self, num_users, num_items, 
                 embedding_size=64, mlp_layers=[128, 64, 32], dropout=0.3):
        super().__init__()
        
        # Embeddings
        self.user_gmf_emb = nn.Embedding(num_users, embedding_size)
        self.item_gmf_emb = nn.Embedding(num_items, embedding_size)
        self.user_mlp_emb = nn.Embedding(num_users, embedding_size)
        self.item_mlp_emb = nn.Embedding(num_items, embedding_size)
        
        # MLP
        mlp = []
        input_dim = embedding_size * 2
        for output_dim in mlp_layers:
            mlp.extend([
                nn.Linear(input_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = output_dim
        self.mlp = nn.Sequential(*mlp)
        
        # Output
        self.output = nn.Linear(embedding_size + mlp_layers[-1], 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for emb in [self.user_gmf_emb, self.item_gmf_emb, 
                   self.user_mlp_emb, self.item_mlp_emb]:
            nn.init.xavier_uniform_(emb.weight)
        
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        
        nn.init.xavier_uniform_(self.output.weight)
    
    def forward(self, user, item):
        # GMF
        gmf_user = self.user_gmf_emb(user)
        gmf_item = self.item_gmf_emb(item)
        gmf = gmf_user * gmf_item
        
        # MLP
        mlp_user = self.user_mlp_emb(user)
        mlp_item = self.item_mlp_emb(item)
        mlp = self.mlp(torch.cat([mlp_user, mlp_item], dim=1))
        
        # Concatenate and predict
        pred = self.output(torch.cat([gmf, mlp], dim=1))
        return torch.sigmoid(pred).squeeze()


# ============= Training Function =============

def train_model(model, train_loader, val_loader, num_epochs=30, lr=0.001, patience=7, device='cpu'):
    """Train the NCF model with early stopping"""
    model = model.to(device)
    # Binary Cross Entropy Loss
    criterion = nn.BCELoss()
    # Adam Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        train_losses = []

        for users, items, labels in tqdm(train_loader, desc=f"Training epoch {epoch+1}/{num_epochs}"):
            users = users.to(device)
            items = items.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(users, items)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # Validation step
        model.eval()
        val_losses = []

        with torch.no_grad():
            for users, items, labels in val_loader:
                users = users.to(device)
                items = items.to(device)
                labels = labels.to(device)

                outputs = model(users, items)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "ncf_model.pt") 
            print(f"Model saved (val loss: {best_val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break

    print("Training complete.")


# ============= Evaluation Function =============

def evaluate(model, test_df, device, k=10):
    """Evaluate the model using Recall@k and NDCG@k"""
    print(f"Evaluating model with k={k}...")
    
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
    
    mean_recall = np.mean(recalls)
    mean_ndcg = np.mean(ndcgs)
    print(f"Recall@{k}: {mean_recall:.4f}, NDCG@{k}: {mean_ndcg:.4f}")
    
    return mean_recall, mean_ndcg


# ============= Main Function =============

def main():
    """Main function to run the complete movie recommendation pipeline"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Step 1: Convert DAT files to CSV (optional - run only if needed)
    convert_dat_files = input("Convert DAT files to CSV? (y/n): ").lower() == 'y'
    if convert_dat_files:
        convert_dat_to_csv()
    
    # Step 2: Preprocess data (optional - run only if needed)
    preprocess_data = input("Preprocess ratings data? (y/n): ").lower() == 'y'
    if preprocess_data:
        df = preprocess_ratings()
        full_df = generate_negatives(df)
        train_df, val_df, test_df = split_dataset(full_df)
    else:
        # Load preprocessed data
        train_df = pd.read_csv("./dataset/train.csv")
        val_df = pd.read_csv("./dataset/val.csv")
        test_df = pd.read_csv("./dataset/test.csv")
    
    # Step 3: Train model
    train_model_flag = input("Train model? (y/n): ").lower() == 'y'
    if train_model_flag:
        # Prepare datasets and dataloaders
        train_set = NCFDataset(train_df)
        val_set = NCFDataset(val_df)
        batch_size = 1024
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)

        # Model initialization
        num_users = train_df["userId"].max() + 1
        num_items = train_df["movie_idx"].max() + 1
        model = NCF(num_users, num_items).to(device)
        
        # Training configuration
        lr = 0.0005
        patience = 5
        num_epochs = 30
        
        # Train model
        train_model(model, train_loader, val_loader, num_epochs=num_epochs, 
                   lr=lr, patience=patience, device=device)
    
    # Step 4: Evaluate model
    evaluate_model = input("Evaluate model? (y/n): ").lower() == 'y'
    if evaluate_model:
        num_users = test_df["userId"].max() + 1
        num_items = test_df["movie_idx"].max() + 1
        model = NCF(num_users, num_items).to(device)
        model.load_state_dict(torch.load("ncf_model.pt", map_location=device))
        evaluate(model, test_df, device, k=10)


if __name__ == "__main__":
    main()