# for my machine (window)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_results(experiment_dir=None):
    """
    Analyze and visualize ablation study results
    
    Parameters:
    - experiment_dir: Directory containing results.json (if results not provided directly)
    """
    # Handle directory finding
    if experiment_dir is None:
        # Find the most recent experiment directory
        dirs = [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('ablation_')]
        if not dirs:
            print("No ablation study results found!")
            return
        experiment_dir = max(dirs)  # Most recent by name
    
    # Check if directory exists
    if not os.path.exists(experiment_dir):
        print(f"Directory {experiment_dir} not found!")
        return
        
    results_path = os.path.join(experiment_dir, "results.json")
    if not os.path.exists(results_path):
        print(f"No results found in {experiment_dir}")
        return
        
    print(f"Analyzing ablation study results from: {experiment_dir}")
    
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Create output directory for analysis
    analysis_dir = os.path.join(experiment_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Save summary to CSV
    summary_path = os.path.join(analysis_dir, "summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")
    
    # Analysis: Best configurations
    print("\n=== Best Configurations ===")
    best_recall = df.loc[df["recall@10"].idxmax()]
    best_ndcg = df.loc[df["ndcg@10"].idxmax()]
    
    print(f"Best Recall@10: {best_recall['recall@10']:.4f} - Config: {best_recall['experiment_name']}")
    print(f"Best NDCG@10: {best_ndcg['ndcg@10']:.4f} - Config: {best_ndcg['experiment_name']}")
    
    # Save detailed best configs
    with open(os.path.join(analysis_dir, "best_configs.txt"), 'w') as f:
        f.write("=== Best Recall@10 Configuration ===\n")
        for key, value in best_recall.items():
            f.write(f"{key}: {value}\n")
        f.write("\n=== Best NDCG@10 Configuration ===\n")
        for key, value in best_ndcg.items():
            f.write(f"{key}: {value}\n")
    
    # Create plots for each hyperparameter
    plot_parameters(df, analysis_dir)
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Between Parameters and Metrics")
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "correlation_heatmap.png"))
    print(f"Correlation heatmap saved to {os.path.join(analysis_dir, 'correlation_heatmap.png')}")
    
    # Create tradeoff plot: Recall vs Training Time
    plt.figure(figsize=(10, 6))
    plt.scatter(df["training_time"], df["recall@10"], alpha=0.7)
    for i, row in df.iterrows():
        plt.annotate(f"emb{row['embedding_size']}_lr{row['learning_rate']}", 
                    (row["training_time"], row["recall@10"]),
                    fontsize=8)
    plt.xlabel("Training Time (seconds)")
    plt.ylabel("Recall@10")
    plt.title("Performance vs. Computational Cost")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(analysis_dir, "performance_vs_cost.png"))
    print(f"Performance vs. cost plot saved to {os.path.join(analysis_dir, 'performance_vs_cost.png')}")
    
    print(f"\nAnalysis complete. All visualizations saved to {analysis_dir}")
    return df

def plot_parameters(df, output_dir):
    """Create plots for each hyperparameter to show their impact on metrics"""
    continuous_params = ["embedding_size", "learning_rate", "dropout", "neg_samples"]
    
    # Plot impact of each continuous parameter
    for param in continuous_params:
        unique_values = sorted(df[param].unique())
        if len(unique_values) > 1:
            # We can create line plots for these
            plt.figure(figsize=(12, 6))
            
            # Group by the parameter and compute mean and std
            grouped = df.groupby(param)[["recall@10", "ndcg@10"]].agg(['mean', 'std']).reset_index()
            
            # Plot lines for recall and ndcg
            for metric in ["recall@10", "ndcg@10"]:
                means = grouped[(metric, 'mean')]
                stds = grouped[(metric, 'std')]
                
                plt.errorbar(
                    unique_values, means, yerr=stds, 
                    marker='o', capsize=4, label=metric
                )
            
            plt.xlabel(param)
            plt.ylabel("Performance Metric")
            plt.title(f"Impact of {param} on Model Performance")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.savefig(os.path.join(output_dir, f"{param}_impact.png"))
            print(f"Parameter impact plot for {param} saved")
    
    # Handle batch_size (categorical)
    if len(df["batch_size"].unique()) > 1:
        plt.figure(figsize=(12, 6))
        batch_groups = df.groupby("batch_size")[["recall@10", "ndcg@10"]].mean().reset_index()
        
        # Bar plot
        x = np.arange(len(batch_groups))
        width = 0.35
        
        plt.bar(x - width/2, batch_groups["recall@10"], width, label="Recall@10")
        plt.bar(x + width/2, batch_groups["ndcg@10"], width, label="NDCG@10")
        
        plt.xlabel("Batch Size")
        plt.ylabel("Performance Metric")
        plt.title("Impact of Batch Size on Model Performance")
        plt.xticks(x, batch_groups["batch_size"])
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(output_dir, "batch_size_impact.png"))
        print(f"Batch size impact plot saved")
    
    # Handle MLP Layers (categorical, need special treatment)
    if len(df["experiment_name"].str.extract(r'mlp([0-9_]+)').dropna()[0].unique()) > 1:
        plt.figure(figsize=(14, 6))
        
        # Extract MLP architectures as strings for grouping
        df["mlp_architecture"] = df["experiment_name"].str.extract(r'mlp([0-9_]+)')[0]
        mlp_groups = df.groupby("mlp_architecture")[["recall@10", "ndcg@10"]].mean().reset_index()
        
        # Bar plot
        x = np.arange(len(mlp_groups))
        width = 0.35
        
        plt.bar(x - width/2, mlp_groups["recall@10"], width, label="Recall@10")
        plt.bar(x + width/2, mlp_groups["ndcg@10"], width, label="NDCG@10")
        
        plt.xlabel("MLP Architecture")
        plt.ylabel("Performance Metric")
        plt.title("Impact of MLP Layers on Model Performance")
        plt.xticks(x, mlp_groups["mlp_architecture"], rotation=45)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "mlp_layers_impact.png"))
        print(f"MLP layers impact plot saved")

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Analyze NCF Ablation Study Results')
    parser.add_argument('--dir', type=str, help='Directory containing ablation study results', default=None)
    parser.add_argument('--list', action='store_true', help='List available ablation study directories')
    args = parser.parse_args()
    
    if args.list:
        # List available ablation studies
        dirs = [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('ablation_')]
        if not dirs:
            print("No ablation study results found!")
            return
            
        print("\nAvailable ablation studies:")
        for i, d in enumerate(sorted(dirs, reverse=True)):
            results_path = os.path.join(d, "results.json")
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    results = json.load(f)
                print(f"{i+1}. {d} ({len(results)} experiments)")
            else:
                print(f"{i+1}. {d} (no results.json found)")
        return
        
    # Analyze results
    analyze_results(args.dir)

if __name__ == "__main__":
    main()