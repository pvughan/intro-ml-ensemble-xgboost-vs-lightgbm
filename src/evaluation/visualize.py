"""
Enhanced visualization module with scalability analysis charts
Generates plots for multi-dataset performance comparison
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path


def plot_scalability_comparison(consolidated_csv="scalability_results.csv", output_dir="plots"):
    """
    Generate scalability comparison visualizations across datasets
    
    Args:
        consolidated_csv: Path to consolidated results CSV
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load consolidated results
    df = pd.read_csv(consolidated_csv)
    
    # Set style
    sns.set_style("whitegrid")
    colors = {'XGBoost': '#FF6B6B', 'LightGBM': '#4ECDC4'}
    
    print("\n" + "="*80)
    print("GENERATING SCALABILITY VISUALIZATIONS")
    print("="*80 + "\n")
    
    # 1. Training Time vs Dataset Size
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Average training time by dataset and model
    time_stats = df.groupby(['dataset', 'model', 'dataset_size'])['train_time_sec'].mean().reset_index()
    
    for model in ['XGBoost', 'LightGBM']:
        model_data = time_stats[time_stats['model'] == model]
        ax.plot(model_data['dataset_size'], model_data['train_time_sec'], 
                marker='o', linewidth=2.5, markersize=10, 
                label=model, color=colors[model])
    
    ax.set_xlabel('Dataset Size (samples)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Training Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Training Time Scalability: XGBoost vs LightGBM', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add dataset labels
    for dataset in df['dataset'].unique():
        dataset_data = time_stats[time_stats['dataset'] == dataset].iloc[0]
        ax.annotate(dataset, 
                   xy=(dataset_data['dataset_size'], dataset_data['train_time_sec']),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scalability_training_time.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/scalability_training_time.png")
    plt.close()
    
    # 2. Memory Usage Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    mem_stats = df.groupby(['dataset', 'model'])['memory_increase_mb'].mean().reset_index()
    
    datasets = mem_stats['dataset'].unique()
    x = np.arange(len(datasets))
    width = 0.35
    
    xgb_mem = [mem_stats[(mem_stats['dataset'] == d) & (mem_stats['model'] == 'XGBoost')]['memory_increase_mb'].values[0] 
               for d in datasets]
    lgbm_mem = [mem_stats[(mem_stats['dataset'] == d) & (mem_stats['model'] == 'LightGBM')]['memory_increase_mb'].values[0] 
                for d in datasets]
    
    bars1 = ax.bar(x - width/2, xgb_mem, width, label='XGBoost', color=colors['XGBoost'], alpha=0.8)
    bars2 = ax.bar(x + width/2, lgbm_mem, width, label='LightGBM', color=colors['LightGBM'], alpha=0.8)
    
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Avg Memory Increase (MB)', fontsize=12, fontweight='bold')
    ax.set_title('Memory Usage Comparison Across Datasets', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scalability_memory_usage.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/scalability_memory_usage.png")
    plt.close()
    
    # 3. Accuracy vs Dataset Scale
    fig, ax = plt.subplots(figsize=(12, 7))
    
    acc_stats = df.groupby(['dataset', 'model', 'dataset_size'])['accuracy'].mean().reset_index()
    
    for model in ['XGBoost', 'LightGBM']:
        model_data = acc_stats[acc_stats['model'] == model]
        ax.plot(model_data['dataset_size'], model_data['accuracy'], 
                marker='o', linewidth=2.5, markersize=10, 
                label=model, color=colors[model])
    
    ax.set_xlabel('Dataset Size (samples)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Across Dataset Scales', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add dataset labels
    for dataset in df['dataset'].unique():
        dataset_data = acc_stats[acc_stats['dataset'] == dataset].iloc[0]
        ax.annotate(dataset, 
                   xy=(dataset_data['dataset_size'], dataset_data['accuracy']),
                   xytext=(10, -15), textcoords='offset points',
                   fontsize=9, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scalability_accuracy_trend.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/scalability_accuracy_trend.png")
    plt.close()
    
    # 4. Speedup Factor Analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate speedup (XGBoost time / LightGBM time)
    time_pivot = df.groupby(['dataset', 'model'])['train_time_sec'].mean().unstack()
    
    if 'XGBoost' in time_pivot.columns and 'LightGBM' in time_pivot.columns:
        speedup = time_pivot['XGBoost'] / time_pivot['LightGBM']
        
        bars = ax.bar(range(len(speedup)), speedup.values, color='#95E1D3', alpha=0.8, edgecolor='black')
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='No speedup (1.0x)')
        
        ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
        ax.set_ylabel('Speedup Factor (XGBoost time / LightGBM time)', fontsize=12, fontweight='bold')
        ax.set_title('LightGBM Speedup Over XGBoost', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(speedup)))
        ax.set_xticklabels(speedup.index)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}x',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'scalability_speedup_factor.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_dir}/scalability_speedup_factor.png")
        plt.close()
    
    print("\n✅ All scalability visualizations generated successfully!\n")


def plot_single_dataset_benchmark(csv_path, dataset_name, output_dir="plots"):
    """
    Generate benchmark plots for a single dataset
    (Original visualization function retained for backward compatibility)
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    if 'dataset' in df.columns:
        df = df[df['dataset'] == dataset_name]
    
    metrics_cols = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
    
    # Grouped bar chart for metrics
    fig, ax = plt.subplots(figsize=(12, 6))
    
    xgb_avg = df[df['model'] == 'XGBoost'][metrics_cols].mean()
    lgbm_avg = df[df['model'] == 'LightGBM'][metrics_cols].mean()
    
    x = np.arange(len(metrics_cols))
    width = 0.35
    
    ax.bar(x - width/2, xgb_avg.values, width, label='XGBoost', color='#FF6B6B', alpha=0.8)
    ax.bar(x + width/2, lgbm_avg.values, width, label='LightGBM', color='#4ECDC4', alpha=0.8)
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Average Performance: {dataset_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics_cols])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'benchmark_{dataset_name}.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/benchmark_{dataset_name}.png")
    plt.close()


if __name__ == "__main__":
    # Check for consolidated results
    if os.path.exists("scalability_results.csv"):
        plot_scalability_comparison("scalability_results.csv", "plots")
    
    # Also generate individual dataset plots if available
    for csv_file in Path(".").glob("ablation_results_*.csv"):
        dataset_name = csv_file.stem.replace("ablation_results_", "")
        print(f"\nGenerating plots for {dataset_name}...")
        plot_single_dataset_benchmark(str(csv_file), dataset_name, "plots")
