import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score

class MLDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.users = torch.tensor(self.data['userId'].values, dtype=torch.long)
        self.items = torch.tensor(self.data['movie_idx'].values, dtype=torch.long)
        self.labels = torch.tensor(self.data['label'].values, dtype=torch.float32)  # One-hot encoded to 0 or 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # GMF 
        self.gmf = nn.Linear(embedding_dim, 1)
        
        # MLP
        self.mlp_fc1 = nn.Linear(embedding_dim * 2, 128)
        self.mlp_fc2 = nn.Linear(128, 64)
        self.mlp_fc3 = nn.Linear(64, 32)
        
        # Fusion Layer
        self.final_fc = nn.Linear(33, 1)  # 32 from MLP + 1 from GMF
        self.activation = nn.ReLU()

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        
        # GMF 
        gmf_out = self.gmf(user_emb * item_emb)
        
        # MLP 
        mlp_input = torch.cat([user_emb, item_emb], dim=-1)
        mlp_out = self.activation(self.mlp_fc1(mlp_input))
        mlp_out = self.activation(self.mlp_fc2(mlp_out))
        mlp_out = self.activation(self.mlp_fc3(mlp_out))
        
        # Fusion Layer
        combined = torch.cat([gmf_out, mlp_out], dim=-1)
        output = torch.sigmoid(self.final_fc(combined))
        return output.squeeze()


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    for users, items, labels in train_loader:
        users, items, labels = users.to(device), items.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(users, items)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def evaluate_model(model, val_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for users, items, labels in val_loader:
            users, items = users.to(device), items.to(device)
            outputs = model(users, items)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.numpy())
    recall = np.mean([recall_at_k(all_labels, all_preds, k=10)])
    ndcg = np.mean([ndcg_score([all_labels], [all_preds])])
    return recall, ndcg


def recall_at_k(labels, preds, k):
    top_k = np.argsort(preds)[-k:][::-1]
    hits = sum(1 for i in top_k if labels[i] > 0)
    return hits / min(k, len(labels))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = MLDataset('train.csv')
    val_dataset = MLDataset('val.csv')
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)

    model = NCF(num_users=6040, num_items=3706).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        train_model(model, train_loader, criterion, optimizer, device)
        recall, ndcg = evaluate_model(model, val_loader, device)
        print(f'Epoch {epoch+1}, Recall@10: {recall:.4f}, NDCG@10: {ndcg:.4f}')
