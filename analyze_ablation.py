# for my machine (window)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import time
import pandas as pd
import numpy as np
import torch
from itertools import product
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Import from your merged script - assuming it's named "ncf_merged.py"
# Adjust the import based on your actual filename
from ncf_merged import NCF, NCFDataset, train_model, evaluate

# Set global configurations
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 
                      'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Define hyperparameter search spaces
def get_hyperparameter_grid():
    """Define the hyperparameter grid for ablation studies"""
    hyperparams = {
        "embedding_size": [32, 64, 128],
        "learning_rate": [0.0001, 0.0005, 0.001],
        "batch_size": [256, 512, 1024],
        "dropout": [0.1, 0.3, 0.5],
        "mlp_layers": [
            [64, 32],
            [128, 64, 32],
            [256, 128, 64, 32]
        ],
        "neg_samples": [1, 4, 8]  # Number of negative samples per positive
    }
    return hyperparams

def generate_dataset_with_neg_samples(num_neg_samples):
    """Generate dataset with specified number of negative samples per positive"""
    # Load processed positive interactions
    pos_df = pd.read_csv("./dataset/editing_ratings.csv")
    all_items = set(pos_df["movie_idx"].unique())
    user_items = pos_df.groupby("userId")["movie_idx"].apply(set).to_dict()
    
    print(f"Generating dataset with {num_neg_samples} negative samples per positive interaction")
    
    # Generate negative samples
    negatives = []
    for user in tqdm(user_items):
        pos_items = user_items[user]
        neg_candidates = list(all_items - pos_items)
        neg_samples = np.random.choice(neg_candidates, size=len(pos_items) * num_neg_samples, replace=True)
        
        for item in neg_samples:
            negatives.append([user, item, 0])
    
    # Combine positives and negatives
    df_neg = pd.DataFrame(negatives, columns=["userId", "movie_idx", "label"])
    full_df = pd.concat([pos_df[["userId", "movie_idx", "label"]], df_neg])
    
    # Shuffle dataset
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into train, val, test
    train_df, temp_df = train_test_split(
        full_df, test_size=0.3, stratify=full_df["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
    )
    
    # Get dataset sizes
    num_users = full_df["userId"].max() + 1
    num_items = full_df["movie_idx"].max() + 1
    
    return train_df, val_df, test_df, num_users, num_items

def load_data():
    """Load preprocessed data for ablation studies - uses default 8 neg samples"""
    train_df = pd.read_csv("./dataset/train.csv")
    val_df = pd.read_csv("./dataset/val.csv")
    test_df = pd.read_csv("./dataset/test.csv")
    
    # Get dataset sizes
    num_users = max(train_df["userId"].max(), val_df["userId"].max(), test_df["userId"].max()) + 1
    num_items = max(train_df["movie_idx"].max(), val_df["movie_idx"].max(), test_df["movie_idx"].max()) + 1
    
    return train_df, val_df, test_df, num_users, num_items

def run_single_experiment(param_config, train_df, val_df, test_df, num_users, num_items, experiment_dir, param_names):
    """Run a single experiment with the given hyperparameter configuration"""
    # Extract parameters
    embedding_size = param_config["embedding_size"]
    learning_rate = param_config["learning_rate"]
    batch_size = param_config["batch_size"]
    dropout = param_config["dropout"]
    mlp_layers = param_config["mlp_layers"]
    neg_samples = param_config.get("neg_samples", 8)  # Default to 8 if not specified
    
    # Check if we need to regenerate the dataset with different negative samples
    if "neg_samples" in param_config and neg_samples != 8:
        train_df, val_df, test_df, num_users, num_items = generate_dataset_with_neg_samples(neg_samples)
    
    # Create experiment name - only include the parameters being studied in the name
    name_parts = []
    if "embedding_size" in param_names:
        name_parts.append(f"emb{embedding_size}")
    if "learning_rate" in param_names:
        name_parts.append(f"lr{learning_rate}")
    if "batch_size" in param_names:
        name_parts.append(f"batch{batch_size}")
    if "dropout" in param_names:
        name_parts.append(f"drop{dropout}")
    if "neg_samples" in param_names:
        name_parts.append(f"neg{neg_samples}")
    if "mlp_layers" in param_names:
        name_parts.append(f"mlp{'_'.join([str(x) for x in mlp_layers])}")
        
    experiment_name = "_".join(name_parts)
    
    print(f"Running experiment: {experiment_name}")
    
    # Create dataloaders
    train_set = NCFDataset(train_df)
    val_set = NCFDataset(val_df)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    
    # Initialize model
    model = NCF(num_users, num_items, 
                embedding_size=embedding_size, 
                mlp_layers=mlp_layers, 
                dropout=dropout).to(device)
    
    # Set up model checkpointing
    model_save_path = os.path.join(experiment_dir, f"{experiment_name}.pt")
    
    # Train the model
    start_time = time.time()
    
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=30,
        lr=learning_rate,
        patience=5,
        device=device,
    )
    
    # Move best model checkpoint to the experiment dir
    if os.path.exists("ncf_model.pt"):
        os.rename("ncf_model.pt", model_save_path)
    
    training_time = time.time() - start_time
    
    # Load the best model for evaluation
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    
    # Evaluate on test set
    recall, ndcg = evaluate(model, test_df, device, k=10)
    
    # Record results
    results = {
        "experiment_name": experiment_name,
        "embedding_size": embedding_size,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "dropout": dropout,
        "neg_samples": neg_samples,
        "mlp_layers": mlp_layers,
        "recall@10": float(recall),
        "ndcg@10": float(ndcg),
        "training_time": training_time,
        "model_path": model_save_path
    }
    
    return results

def run_ablation_study(selected_params):
    """
    Run ablation study on specified parameters
    
    Parameters:
    - selected_params: dict of parameter names to study
    """
    # Create directory name based on the parameters being studied
    param_name_short = '_'.join([p[:3] for p in selected_params])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"ablation_{param_name_short}_{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Load data
    train_df, val_df, test_df, num_users, num_items = load_data()
    
    # Get hyperparameter grid
    hyperparams = get_hyperparameter_grid()
    
    # If selected params provided, only vary those
    if selected_params:
        # Create a default config first
        default_config = {
            "embedding_size": 64,
            "learning_rate": 0.0005,
            "batch_size": 1024,
            "dropout": 0.3,
            "mlp_layers": [128, 64, 32],
            "neg_samples": 8
        }
        
        experiments = []
        
        # For each selected parameter, vary it while keeping others at default
        for param_name in selected_params:
            if param_name in hyperparams:
                for param_value in hyperparams[param_name]:
                    config = default_config.copy()
                    config[param_name] = param_value
                    experiments.append(config)
    else:
        # Full grid search (WARNING: This can be very computationally expensive)
        param_names = hyperparams.keys()
        param_values = [hyperparams[name] for name in param_names]
        experiments = [dict(zip(param_names, values)) for values in product(*param_values)]
    
    print(f"Running {len(experiments)} experiments")
    
    # Track all results
    all_results = []
    
    # Run experiments
    for i, param_config in enumerate(experiments):
        print(f"\nExperiment {i+1}/{len(experiments)}")
        result = run_single_experiment(
            param_config, train_df, val_df, test_df, 
            num_users, num_items, experiment_dir, selected_params
        )
        all_results.append(result)
        
        # Save results after each experiment
        results_path = os.path.join(experiment_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    print(f"Ablation study complete. Results saved to {experiment_dir}/results.json")
    return experiment_dir

def main():
    print("NCF Hyperparameter Ablation Study")
    print("=================================")
    
    # Run focused study
    print("\nSelect parameters to vary in the ablation study:")
    print("1. Embedding Size")
    print("2. Learning Rate")
    print("3. Batch Size")
    print("4. Dropout")
    print("5. MLP Layers")
    print("6. Number of Negative Samples")
    
    selected = input("Enter parameter numbers separated by spaces (e.g., '1 3 4'): ")
    param_mapping = {
        "1": "embedding_size",
        "2": "learning_rate",
        "3": "batch_size",
        "4": "dropout",
        "5": "mlp_layers",
        "6": "neg_samples"
    }
    
    selected_params = [param_mapping[x] for x in selected.split() if x in param_mapping]
    
    if not selected_params:
        print("No valid parameters selected. Exiting.")
        return
        
    print(f"Running ablation study on: {', '.join(selected_params)}")
    experiment_dir = run_ablation_study(selected_params)
    print(f"\nTo analyze the results, run the analysis script:")
    print(f"python analyze_ablation.py --dir {experiment_dir}")

if __name__ == "__main__":
    main()