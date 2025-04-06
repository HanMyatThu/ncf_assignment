import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import argparse
import os

def load_ablation_results(file_path):
    """Load the ablation study results from JSON file"""
    with open(file_path, 'r') as f:
        results = json.load(f)
    return results

def extract_parameter_type(experiment_name):
    """Extract the parameter type from experiment name"""
    if experiment_name.startswith('emb'):
        return 'embedding_size'
    elif experiment_name.startswith('lr'):
        return 'learning_rate'
    elif experiment_name.startswith('batch'):
        return 'batch_size'
    elif experiment_name.startswith('drop'):
        return 'dropout'
    elif experiment_name.startswith('neg'):
        return 'neg_samples'
    elif experiment_name.startswith('mlp'):
        return 'mlp_layers'
    else:
        return 'unknown'

def extract_parameter_value(experiment_name, param_type):
    """Extract the parameter value from experiment name"""
    if param_type == 'embedding_size':
        return int(experiment_name.replace('emb', ''))
    elif param_type == 'learning_rate':
        return float(experiment_name.replace('lr', ''))
    elif param_type == 'batch_size':
        return int(experiment_name.replace('batch', ''))
    elif param_type == 'dropout':
        return float(experiment_name.replace('drop', ''))
    elif param_type == 'neg_samples':
        return int(experiment_name.replace('neg', ''))
    elif param_type == 'mlp_layers':
        return experiment_name.replace('mlp', '')
    else:
        return experiment_name

def prepare_data(results):
    """Transform results into a DataFrame suitable for plotting"""
    # Extract parameter types and values
    for result in results:
        result['param_type'] = extract_parameter_type(result['experiment_name'])
        result['param_value'] = extract_parameter_value(result['experiment_name'], result['param_type'])
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    return df

def plot_metric_comparison(df, output_dir):
    """Plot Recall@10 vs NDCG@10 for all experiments"""
    plt.figure(figsize=(10, 6))
    
    # Create a color map based on parameter type
    param_types = df['param_type'].unique()
    colors = plt.cm.get_cmap('tab10', len(param_types))
    
    # Create a scatter plot
    for i, param_type in enumerate(param_types):
        subset = df[df['param_type'] == param_type]
        plt.scatter(subset['recall@10'], subset['ndcg@10'], 
                    label=param_type, 
                    color=colors(i), 
                    s=80, 
                    alpha=0.7)
        
        # Add experiment names as annotations
        for _, row in subset.iterrows():
            plt.annotate(row['experiment_name'], 
                        (row['recall@10'], row['ndcg@10']),
                        fontsize=8,
                        alpha=0.8,
                        xytext=(5, 5),
                        textcoords='offset points')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Recall@10', fontsize=12)
    plt.ylabel('NDCG@10', fontsize=12)
    plt.title('Performance Comparison: Recall vs NDCG', fontsize=14)
    plt.legend(title='Parameter Type')
    
    # Format the axes to show more decimal places
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300)
    plt.close()

def plot_parameter_impact(df, output_dir):
    """Create bar plots showing the impact of each parameter on performance metrics"""
    # Calculate performance impact for each parameter type
    impact_data = []
    
    for param_type in df['param_type'].unique():
        subset = df[df['param_type'] == param_type]
        
        recall_min = subset['recall@10'].min()
        recall_max = subset['recall@10'].max()
        recall_impact = (recall_max - recall_min) / recall_min * 100
        
        ndcg_min = subset['ndcg@10'].min()
        ndcg_max = subset['ndcg@10'].max()
        ndcg_impact = (ndcg_max - ndcg_min) / ndcg_min * 100
        
        avg_impact = (recall_impact + ndcg_impact) / 2
        
        impact_data.append({
            'parameter': param_type,
            'recall_impact': recall_impact,
            'ndcg_impact': ndcg_impact,
            'avg_impact': avg_impact
        })
    
    impact_df = pd.DataFrame(impact_data)
    impact_df = impact_df.sort_values('avg_impact', ascending=False)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(impact_df))
    width = 0.35
    
    plt.bar(x - width/2, impact_df['recall_impact'], width, label='Recall@10', color='#1f77b4', alpha=0.8)
    plt.bar(x + width/2, impact_df['ndcg_impact'], width, label='NDCG@10', color='#ff7f0e', alpha=0.8)
    
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='1% threshold')
    
    plt.xlabel('Hyperparameter', fontsize=12)
    plt.ylabel('Performance Impact (%)', fontsize=12)
    plt.title('Impact of Each Hyperparameter on Model Performance', fontsize=14)
    plt.xticks(x, impact_df['parameter'])
    plt.legend()
    
    # Add values on top of bars
    for i, v in enumerate(impact_df['recall_impact']):
        plt.text(i - width/2, v + 0.1, f'{v:.2f}%', ha='center', fontsize=9)
    
    for i, v in enumerate(impact_df['ndcg_impact']):
        plt.text(i + width/2, v + 0.1, f'{v:.2f}%', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_impact.png'), dpi=300)
    plt.close()

def plot_parameter_performance(df, output_dir):
    """Create detailed plots for each parameter type showing its effect on metrics"""
    # Process each parameter type
    for param_type in df['param_type'].unique():
        subset = df[df['param_type'] == param_type]
        
        # Sort by parameter value if possible
        try:
            subset = subset.sort_values('param_value')
        except:
            # If sorting fails (e.g., for mlp_layers), use as is
            pass
        
        plt.figure(figsize=(10, 6))
        
        x = np.arange(len(subset))
        width = 0.35
        
        # Create the main axis for bars
        ax1 = plt.gca()
        bar1 = ax1.bar(x - width/2, subset['recall@10'], width, label='Recall@10', color='#1f77b4', alpha=0.8)
        bar2 = ax1.bar(x + width/2, subset['ndcg@10'], width, label='NDCG@10', color='#ff7f0e', alpha=0.8)
        
        ax1.set_xlabel(f'{param_type} Value', fontsize=12)
        ax1.set_ylabel('Performance Metric', fontsize=12)
        ax1.set_title(f'Effect of {param_type} on Model Performance', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(subset['experiment_name'])
        
        # Add values on top of bars
        for i, v in enumerate(subset['recall@10']):
            ax1.text(i - width/2, v + 0.005, f'{v:.3f}', ha='center', fontsize=8)
        
        for i, v in enumerate(subset['ndcg@10']):
            ax1.text(i + width/2, v + 0.005, f'{v:.3f}', ha='center', fontsize=8)
        
        # Add secondary y-axis for training time
        ax2 = ax1.twinx()
        line = ax2.plot(x, subset['training_time'], 'g-', marker='o', label='Training Time')
        ax2.set_ylabel('Training Time (seconds)', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        
        # For each point on the line, add the value
        for i, v in enumerate(subset['training_time']):
            ax2.annotate(f'{v:.1f}s', 
                       (i, v),
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center',
                       fontsize=8,
                       color='g')
        
        # Combine legends with proper spacing
        legend_elements = [bar1, bar2, line[0]]
        legend_labels = ['Recall@10', 'NDCG@10', 'Training Time']
        ax1.legend(legend_elements, legend_labels, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{param_type}_performance.png'), dpi=300)
        plt.close()

def plot_training_time_vs_performance(df, output_dir):
    """Create a scatter plot of training time vs performance metrics with parameter type coloring"""
    plt.figure(figsize=(12, 6))
    
    param_types = df['param_type'].unique()
    colors = plt.cm.get_cmap('tab10', len(param_types))
    
    # Create two subplots: one for Recall, one for NDCG
    plt.subplot(1, 2, 1)
    for i, param_type in enumerate(param_types):
        subset = df[df['param_type'] == param_type]
        plt.scatter(subset['training_time'], subset['recall@10'], label=param_type, color=colors(i), s=80, alpha=0.7)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Training Time (seconds)', fontsize=10)
    plt.ylabel('Recall@10', fontsize=10)
    plt.title('Training Time vs Recall@10', fontsize=12)
    
    plt.subplot(1, 2, 2)
    for i, param_type in enumerate(param_types):
        subset = df[df['param_type'] == param_type]
        plt.scatter(subset['training_time'], subset['ndcg@10'], label=param_type, color=colors(i), s=80, alpha=0.7)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Training Time (seconds)', fontsize=10)
    plt.ylabel('NDCG@10', fontsize=10)
    plt.title('Training Time vs NDCG@10', fontsize=12)
    
    # Single legend for both subplots - positioned lower to avoid x-axis label
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='lower center', ncol=len(param_types), title='Parameter Type', 
                 bbox_to_anchor=(0.5, -0.15))  # Moved lower
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjusted to make room for the legend at the bottom
    plt.savefig(os.path.join(output_dir, 'training_time_vs_performance.png'), dpi=300, bbox_inches='tight')  # Added bbox_inches to ensure legend is included
    plt.close()

def plot_heatmap(df, output_dir):
    """Create a correlation heatmap showing relationships between parameters and metrics"""
    # Convert experiment_name to numeric codes
    df['experiment_code'] = df['experiment_name'].astype('category').cat.codes
    
    # Select columns for correlation
    numeric_columns = ['embedding_size', 'learning_rate', 'batch_size', 'dropout', 
                       'neg_samples', 'recall@10', 'ndcg@10', 'training_time']
    
    # Create correlation matrix
    corr_df = df[numeric_columns].corr()
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    sns.heatmap(corr_df, mask=mask, cmap='coolwarm', vmax=.8, vmin=-.8, center=0,
                square=True, linewidths=.5, annot=True, fmt='.2f', cbar_kws={"shrink": .5})
    
    plt.title('Correlation Heatmap', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300)
    plt.close()

def plot_best_config(df, output_dir):
    """Create a bar chart comparing default and best configuration"""
    # Find the experiment with the highest recall@10
    best_exp = df.loc[df['recall@10'].idxmax()]
    
    # Get default values
    default_config = {
        'embedding_size': 64,
        'learning_rate': 0.0005,
        'batch_size': 1024,
        'dropout': 0.3,
        'neg_samples': 8,
    }
    
    # Get best values for each parameter
    best_params = {}
    for param_type in df['param_type'].unique():
        if param_type != 'mlp_layers':  # Skip MLP layers for this visualization
            best_for_param = df[df['param_type'] == param_type].loc[df[df['param_type'] == param_type]['recall@10'].idxmax()]
            best_params[param_type] = best_for_param[param_type]
    
    # Create a comparative bar chart instead of a radar chart
    fig, axs = plt.subplots(len(best_params), 1, figsize=(10, 2*len(best_params)), sharex=False)
    
    # If there's only one parameter, axs won't be an array
    if len(best_params) == 1:
        axs = [axs]
    
    # Create a bar chart for each parameter
    for i, (param, best_val) in enumerate(best_params.items()):
        default_val = default_config[param]
        
        # For some parameters, we need to format the display differently
        if param == 'learning_rate':
            default_label = f"{default_val:.4f}"
            best_label = f"{best_val:.4f}"
        else:
            default_label = f"{default_val}"
            best_label = f"{best_val}"
        
        # Create a bar chart
        bars = axs[i].bar(
            ['Default', 'Best'], 
            [default_val, best_val],
            color=['blue', 'red'],
            alpha=0.7
        )
        
        # Add parameter name as title
        axs[i].set_title(f"{param}")
        
        # Add values on top of bars
        for bar, label in zip(bars, [default_label, best_label]):
            height = bar.get_height()
            axs[i].text(
                bar.get_x() + bar.get_width()/2,
                height * 1.05,
                label,
                ha='center',
                fontsize=10
            )
    
    plt.suptitle('Default vs Best Configuration by Parameter', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the title
    plt.savefig(os.path.join(output_dir, 'config_comparison.png'), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='NCF Ablation Study Visualization')
    parser.add_argument('--input', default='all_ablation_results.json', 
                        help='Path to the ablation results JSON file')
    parser.add_argument('--output', default='visualization_results', 
                        help='Directory to save visualization results')
    parser.add_argument('--skip-radar', action='store_true', 
                        help='Skip the radar chart that might cause errors')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load and prepare data
    results = load_ablation_results(args.input)
    df = prepare_data(results)
    
    # Generate all plots
    print("Generating plots...")
    
    print("1. Plotting metric comparison")
    plot_metric_comparison(df, args.output)
    
    print("2. Plotting parameter impact analysis")
    plot_parameter_impact(df, args.output)
    
    print("3. Plotting individual parameter performance")
    plot_parameter_performance(df, args.output)
    
    print("4. Plotting training time vs performance")
    plot_training_time_vs_performance(df, args.output)
    
    print("5. Plotting correlation heatmap")
    try:
        plot_heatmap(df, args.output)
    except Exception as e:
        print(f"Warning: Could not create correlation heatmap: {str(e)}")
    
    print("6. Plotting configuration comparison")
    try:
        plot_best_config(df, args.output)
    except Exception as e:
        print(f"Warning: Could not create config comparison: {str(e)}")
    
    print(f"All visualizations saved to {args.output}")
    print("Done!")

if __name__ == "__main__":
    main()