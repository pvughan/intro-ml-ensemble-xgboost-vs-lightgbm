"""
Scalability analysis across multiple datasets
Compares XGBoost vs LightGBM performance at different scales
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.loader import load_dataset, get_dataset_info, list_available_datasets
from evaluation.ablation import run_ablation, compare_models_summary
from config import DATASETS


def run_multi_dataset_analysis(dataset_names=None, higgs_sample_size=None):
    """
    Run ablation study across multiple datasets for scalability analysis
    
    Args:
        dataset_names: List of dataset names to analyze. 
                      If None, uses all available datasets.
        higgs_sample_size: Optional sample size for HIGGS dataset 
                          (e.g., 1000000 for 1M samples)
    
    Returns:
        Dictionary of DataFrames, one per dataset
    """
    if dataset_names is None:
        dataset_names = list(DATASETS.keys())
    
    print("\n" + "="*80)
    print("MULTI-DATASET SCALABILITY ANALYSIS")
    print("XGBoost vs LightGBM Performance Comparison")
    print("="*80)
    
    results = {}
    
    for dataset_name in dataset_names:
        print(f"\n{'#'*80}")
        print(f"# DATASET: {dataset_name.upper()}")
        print(f"{'#'*80}\n")
        
        # Load dataset
        try:
            if dataset_name == "higgs" and higgs_sample_size:
                X_train, X_test, y_train, y_test = load_dataset(
                    dataset_name, 
                    sample_size=higgs_sample_size
                )
            else:
                X_train, X_test, y_train, y_test = load_dataset(dataset_name)
            
            # Run ablation study
            df_results = run_ablation(
                X_train, X_test, y_train, y_test,
                dataset_name=dataset_name,
                save_results=True
            )
            
            # Store results
            results[dataset_name] = df_results
            
            # Print summary
            print(f"\n📊 SUMMARY FOR {dataset_name.upper()}:")
            print("-" * 70)
            summary = compare_models_summary(df_results)
            print(summary)
            print("\n")
            
        except FileNotFoundError as e:
            print(f"⚠ Skipping {dataset_name}: {e}")
            continue
        except KeyboardInterrupt:
            print(f"\n⚠ Analysis interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error with {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate consolidated report
    if results:
        generate_consolidated_report(results)
    
    return results


def generate_consolidated_report(results_dict):
    """
    Generate consolidated scalability report across all datasets
    
    Args:
        results_dict: Dictionary of DataFrames from run_multi_dataset_analysis
    """
    print("\n" + "="*80)
    print("CONSOLIDATED SCALABILITY REPORT")
    print("="*80 + "\n")
    
    # Combine all results
    all_results = []
    for dataset_name, df in results_dict.items():
        all_results.append(df)
    
    consolidated_df = pd.concat(all_results, ignore_index=True)
    
    # Save consolidated results
    consolidated_file = "scalability_results.csv"
    consolidated_df.to_csv(consolidated_file, index=False)
    print(f"✓ Consolidated results saved to: {consolidated_file}\n")
    
    # Generate summary statistics
    print("="*80)
    print("AVERAGE PERFORMANCE BY DATASET AND MODEL")
    print("="*80 + "\n")
    
    summary = consolidated_df.groupby(['dataset', 'model']).agg({
        'dataset_size': 'first',
        'train_time_sec': 'mean',
        'accuracy': 'mean',
        'f1': 'mean',
        'roc_auc': 'mean',
        'memory_increase_mb': 'mean'
    }).round(4)
    
    print(summary)
    print("\n")
    
    # Speedup analysis
    print("="*80)
    print("TRAINING TIME COMPARISON (XGBoost vs LightGBM)")
    print("="*80 + "\n")
    
    time_comparison = consolidated_df.groupby(['dataset', 'model'])['train_time_sec'].mean().unstack()
    
    if 'XGBoost' in time_comparison.columns and 'LightGBM' in time_comparison.columns:
        time_comparison['Speedup (XGB/LGBMSpeedup)'] = (
            time_comparison['XGBoost'] / time_comparison['LightGBM']
        ).round(2)
        print(time_comparison)
        print("\n📈 Speedup > 1.0 means LightGBM is faster\n")
    else:
        print(time_comparison)
    
    # Accuracy comparison
    print("="*80)
    print("ACCURACY COMPARISON")
    print("="*80 + "\n")
    
    acc_comparison = consolidated_df.groupby(['dataset', 'model'])['accuracy'].mean().unstack()
    if 'XGBoost' in acc_comparison.columns and 'LightGBM' in acc_comparison.columns:
        acc_comparison['Difference (XGB-LGBM)'] = (
            acc_comparison['XGBoost'] - acc_comparison['LightGBM']
        ).round(4)
    
    print(acc_comparison)
    print("\n")
    
    # Key insights
    print("="*80)
    print("KEY INSIGHTS")
    print("="*80 + "\n")
    
    for dataset_name in consolidated_df['dataset'].unique():
        dataset_data = consolidated_df[consolidated_df['dataset'] == dataset_name]
        
        xgb_time = dataset_data[dataset_data['model'] == 'XGBoost']['train_time_sec'].mean()
        lgbm_time = dataset_data[dataset_data['model'] == 'LightGBM']['train_time_sec'].mean()
        
        xgb_acc = dataset_data[dataset_data['model'] == 'XGBoost']['accuracy'].mean()
        lgbm_acc = dataset_data[dataset_data['model'] == 'LightGBM']['accuracy'].mean()
        
        dataset_size = dataset_data['dataset_size'].iloc[0]
        
        print(f"📊 {dataset_name.upper()} ({dataset_size:,} samples):")
        print(f"   Training Time: XGBoost={xgb_time:.2f}s, LightGBM={lgbm_time:.2f}s "
              f"(Speedup: {xgb_time/lgbm_time:.2f}x)")
        print(f"   Accuracy: XGBoost={xgb_acc:.4f}, LightGBM={lgbm_acc:.4f} "
              f"(Diff: {xgb_acc-lgbm_acc:+.4f})")
        print()
    
    print("="*80 + "\n")
    
    return consolidated_df


if __name__ == "__main__":
    # Example usage
    print("🚀 Starting Multi-Dataset Scalability Analysis\n")
    
    # Option 1: Run on all datasets (this will take time for HIGGS!)
    # results = run_multi_dataset_analysis()
    
    # Option 2: Run on specific datasets
    # results = run_multi_dataset_analysis(['breast_cancer', 'creditcard_fraud'])
    
    # Option 3: Run on all with HIGGS subset
    results = run_multi_dataset_analysis(higgs_sample_size=100000)  # Use 100K samples for HIGGS
    
    print("\n✅ Analysis complete!")
