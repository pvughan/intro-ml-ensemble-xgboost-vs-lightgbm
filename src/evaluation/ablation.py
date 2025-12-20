"""
Enhanced ablation study with timing and memory tracking
Supports multi-dataset scalability analysis
"""

from models.xgboost_model import get_xgboost_model
from models.lightgbm_model import get_lightgbm_model
from evaluation.evaluator import evaluate_model
import pandas as pd
import time
import tracemalloc
import psutil
import os


def get_memory_usage_mb():
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def run_ablation(X_train, X_test, y_train, y_test, dataset_name="unknown", save_results=True):
    """
    Run ablation study with performance tracking
    
    Args:
        X_train, X_test, y_train, y_test: Train/test splits
        dataset_name: Name of dataset for result tracking
        save_results: Whether to save results to CSV
    
    Returns:
        DataFrame with results including timing and memory metrics
    """
    search_space = {
        "learning_rate": [0.01, 0.1],
        "n_estimators": [100, 300]
    }

    records = []
    total_experiments = len(search_space["learning_rate"]) * len(search_space["n_estimators"]) * 2
    experiment_num = 0

    print(f"\n{'='*70}")
    print(f"ABLATION STUDY: {dataset_name}")
    print(f"Total experiments: {total_experiments}")
    print(f"{'='*70}\n")

    for lr in search_space["learning_rate"]:
        for n in search_space["n_estimators"]:

            for model_name, builder in {
                "XGBoost": get_xgboost_model,
                "LightGBM": get_lightgbm_model
            }.items():
                experiment_num += 1
                
                print(f"[{experiment_num}/{total_experiments}] {model_name} | lr={lr}, n_estimators={n}")
                
                # Track memory before training
                mem_before = get_memory_usage_mb()
                
                # Start timing and memory tracking
                tracemalloc.start()
                start_time = time.time()
                
                # Build and train model
                params = {"learning_rate": lr, "n_estimators": n}
                model = builder(params)
                model.fit(X_train, y_train)
                
                # End timing
                train_time = time.time() - start_time
                
                # Get peak memory usage during training
                current_mem, peak_mem = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                peak_mem_mb = peak_mem / (1024 * 1024)
                
                # Memory after training
                mem_after = get_memory_usage_mb()
                mem_increase = mem_after - mem_before
                
                # Evaluate model
                eval_start = time.time()
                metrics = evaluate_model(model, X_test, y_test)
                eval_time = time.time() - eval_start
                
                # Print summary
                print(f"   ✓ Train time: {train_time:.2f}s | Eval time: {eval_time:.3f}s | "
                      f"Memory: +{mem_increase:.1f}MB | Accuracy: {metrics['accuracy']:.4f}")

                # Store results
                records.append({
                    "dataset": dataset_name,
                    "dataset_size": len(y_train) + len(y_test),
                    "train_size": len(y_train),
                    "test_size": len(y_test),
                    "model": model_name,
                    "learning_rate": lr,
                    "n_estimators": n,
                    "train_time_sec": train_time,
                    "eval_time_sec": eval_time,
                    "total_time_sec": train_time + eval_time,
                    "memory_increase_mb": mem_increase,
                    "peak_memory_mb": peak_mem_mb,
                    **metrics
                })

    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Save results
    if save_results:
        results_file = f"ablation_results_{dataset_name}.csv"
        df.to_csv(results_file, index=False)
        print(f"\n✓ Results saved to: {results_file}\n")
    
    return df


def compare_models_summary(df):
    """
    Generate summary comparison between XGBoost and LightGBM
    
    Args:
        df: DataFrame from run_ablation
    
    Returns:
        Summary DataFrame
    """
    summary = df.groupby('model').agg({
        'train_time_sec': ['mean', 'min', 'max'],
        'accuracy': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'roc_auc': ['mean', 'std'],
        'memory_increase_mb': ['mean', 'max']
    }).round(4)
    
    return summary