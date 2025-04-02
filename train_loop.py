import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

def train_model(model, train_loader, val_loader, num_epochs=30, lr=0.001, patience=7, device='cpu'):
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

        for users, items, labels in tqdm(train_loader, desc="Training"):
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
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break

    print("Training complete.")
