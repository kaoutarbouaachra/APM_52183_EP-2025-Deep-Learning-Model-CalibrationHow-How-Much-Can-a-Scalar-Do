import json
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import seaborn as sns

def plot_combined_metrics(results_path, save_dir):
    if not os.path.exists(results_path):
        print(f"Error: Results file not found at {results_path}")
        return

    with open(results_path, 'r') as f:
        results = json.load(f)
    
    methods =list(results.keys())
    first_method = methods[0]
    corruption_list= list(results[first_method]['corruptions'].keys())
    
    colors = {
        'ts': 'red', 
        'ets': 'blue', 
        'tva': 'green', 
        'dac': 'purple', 
        'ensemble': 'black'
    }
    
    # Create output directory
    plots_dir = os.path.join(save_dir, 'combined_plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        
    sns.set_theme(style="whitegrid")
    
    print(f"Generating combined plots for {len(corruption_list)} corruptions...")

    for corruption in corruption_list:

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        metrics_to_plot = [
            ('accuracy', 'Accuracy (Higher is Better)'),
            ('adaptive_ece', 'Adaptive ECE (Lower is Better)'),
            ('error', 'Error Rate (Lower is Better)')
        ]
        
        for ax_idx, (metric_key, title) in enumerate(metrics_to_plot):
            ax = axes[ax_idx]
            
            for method in methods:
                severities = [0, 1, 2, 3, 4, 5]
                values = []
                
                # --- Get Severity 0 Value ---
                try:
                    clean_data = results[method]['clean_baseline']
                    if metric_key == 'error':
                        # Error = 1.0 - Accuracy
                        val = 1.0 - clean_data['accuracy']
                    else:
                        val = clean_data[metric_key]
                except KeyError:
                    continue
                values.append(val)
                
                # --- Get Severities 1-5 ---
                for s in range(1, 6):
                    corr_data = results[method]['corruptions'][corruption][str(s)]
                    if metric_key == 'error':
                        val = 1.0 - corr_data['accuracy']
                    else:
                        val = corr_data[metric_key]
                    values.append(val)
                
                # Plot line for this method
                ax.plot(severities, values, marker='o', linewidth=2, 
                        label=method.upper(), color=colors.get(method, 'gray'))
            
            # Subplot Styling
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel("Corruption Severity")
            ax.set_xticks(severities)
            ax.grid(True, alpha=0.3)
            
            # Common Y-label logic or specific limits
            if metric_key == 'accuracy':
                ax.set_ylim(0, 1.0)
            elif metric_key == 'error':
                ax.set_ylim(0, 1.0) # Error is also 0-1
            elif metric_key =='adaptive_ece':
                ax.set_ylim(bottom=0)
                
            # Only put legend on the first plot to avoid clutter
            if ax_idx == 0:
                ax.legend(fontsize=10)

        # Main Figure Title
        plt.suptitle(f"Performance Analysis: {corruption.replace('_', ' ').title()}", fontsize=16)
        plt.tight_layout()
        
        # Save
        save_path = os.path.join(plots_dir, f"{corruption}_combined.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        
    print(f"Done! Plots saved to {plots_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default='./checkpoints/ovadia_plus_results.json', help='Path to results JSON')
    parser.add_argument('--save', type=str, default='./checkpoints', help='Path to save plots')
    args = parser.parse_args()
    
    plot_combined_metrics(args.results, args.save)
