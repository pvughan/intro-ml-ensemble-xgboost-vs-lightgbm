import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_benchmark_comparison(csv_path="ablation_results.csv", output_dir="plots"):
    """
    Generate comprehensive benchmark comparison visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    df = pd.read_csv(csv_path)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Metrics comparison across all configurations
    metrics_cols = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_cols):
        ax = axes[idx]
        
        # Create grouped bar chart
        df_pivot = df.pivot_table(
            values=metric,
            index=['learning_rate', 'n_estimators'],
            columns='model'
        )
        
        df_pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title(f'{metric.upper()} - Hyperparameter Comparison', fontsize=12, fontweight='bold')
        ax.set_xlabel('(Learning Rate, N Estimators)', fontsize=10)
        ax.set_ylabel(metric.capitalize(), fontsize=10)
        ax.legend(title='Model', loc='lower right')
        ax.grid(axis='y', alpha=0.3)
        
        # Set x-tick labels
        labels = [f"({lr}, {n})" for lr, n in df_pivot.index]
        ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Remove extra subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_all_metrics.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/benchmark_all_metrics.png")
    plt.close()
    
    # 2. Radar chart for model comparison (average across hyperparameters)
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Calculate average metrics for each model
    xgb_avg = df[df['model'] == 'XGBoost'][metrics_cols].mean()
    lgbm_avg = df[df['model'] == 'LightGBM'][metrics_cols].mean()
    
    # Number of variables
    categories = [m.upper() for m in metrics_cols]
    N = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
    angles += angles[:1]
    
    # Plot
    xgb_values = xgb_avg.tolist()
    xgb_values += xgb_values[:1]
    ax.plot(angles, xgb_values, 'o-', linewidth=2, label='XGBoost', color='#FF6B6B')
    ax.fill(angles, xgb_values, alpha=0.25, color='#FF6B6B')
    
    lgbm_values = lgbm_avg.tolist()
    lgbm_values += lgbm_values[:1]
    ax.plot(angles, lgbm_values, 'o-', linewidth=2, label='LightGBM', color='#4ECDC4')
    ax.fill(angles, lgbm_values, alpha=0.25, color='#4ECDC4')
    
    # Fix axis to go in the right order
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)
    ax.set_ylim(0.94, 1.0)
    ax.set_title('Average Performance Across All Benchmarks', size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_radar_chart.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/benchmark_radar_chart.png")
    plt.close()
    
    # 3. Hyperparameter sensitivity heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, model in enumerate(['XGBoost', 'LightGBM']):
        ax = axes[idx]
        model_df = df[df['model'] == model]
        
        # Create pivot table for average scores across all metrics
        avg_scores = []
        for lr in model_df['learning_rate'].unique():
            row_scores = []
            for n_est in sorted(model_df['n_estimators'].unique()):
                subset = model_df[(model_df['learning_rate'] == lr) & (model_df['n_estimators'] == n_est)]
                avg_score = subset[metrics_cols].mean().mean()
                row_scores.append(avg_score)
            avg_scores.append(row_scores)
        
        # Create DataFrame for heatmap
        heatmap_data = pd.DataFrame(
            avg_scores,
            index=sorted(model_df['learning_rate'].unique()),
            columns=sorted(model_df['n_estimators'].unique())
        )
        
        sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlOrRd', 
                    ax=ax, cbar_kws={'label': 'Avg Score'}, vmin=0.95, vmax=0.99)
        ax.set_title(f'{model} - Hyperparameter Sensitivity', fontsize=14, fontweight='bold')
        ax.set_xlabel('N Estimators', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_heatmap.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/benchmark_heatmap.png")
    plt.close()
    
    # 4. Best configuration comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Find best configuration for each model
    xgb_best = df[df['model'] == 'XGBoost'].loc[df[df['model'] == 'XGBoost']['accuracy'].idxmax()]
    lgbm_best = df[df['model'] == 'LightGBM'].loc[df[df['model'] == 'LightGBM']['accuracy'].idxmax()]
    
    best_configs = pd.DataFrame([xgb_best, lgbm_best])
    
    x = range(len(metrics_cols))
    width = 0.35
    
    xgb_scores = [xgb_best[m] for m in metrics_cols]
    lgbm_scores = [lgbm_best[m] for m in metrics_cols]
    
    bars1 = ax.bar([i - width/2 for i in x], xgb_scores, width, 
                   label=f'XGBoost (lr={xgb_best["learning_rate"]}, n={int(xgb_best["n_estimators"])})',
                   color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], lgbm_scores, width,
                   label=f'LightGBM (lr={lgbm_best["learning_rate"]}, n={int(lgbm_best["n_estimators"])})',
                   color='#4ECDC4', alpha=0.8)
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Best Configuration Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics_cols])
    ax.legend(fontsize=10)
    ax.set_ylim(0.94, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_best_config.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/benchmark_best_config.png")
    plt.close()
    
    print("\n✅ All benchmark visualizations generated successfully!")
    return best_configs

if __name__ == "__main__":
    # Run from project root or src/evaluation directory
    import sys
    if os.path.exists("src/ablation_results.csv"):
        plot_benchmark_comparison("src/ablation_results.csv", "plots")
    else:
        plot_benchmark_comparison("../ablation_results.csv", "../../plots")

