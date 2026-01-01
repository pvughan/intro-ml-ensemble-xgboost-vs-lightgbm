"""
Main entry point for XGBoost vs LightGBM comparison
Updated to support multiple datasets for scalability analysis
"""

import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import argparse
from data.loader import load_dataset, list_available_datasets, load_data
from models.xgboost_model import get_xgboost_model
from models.lightgbm_model import get_lightgbm_model
from evaluation.evaluator import evaluate_model
from evaluation.ablation import run_ablation, compare_models_summary
from utils.metrics import print_metrics


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='XGBoost vs LightGBM Comparison')
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='breast_cancer',
        choices=['breast_cancer', 'creditcard_fraud', 'higgs'],
        help='Dataset to use for analysis (default: breast_cancer)'
    )
    parser.add_argument(
        '--higgs-sample',
        type=int,
        default=None,
        help='Sample size for HIGGS dataset (e.g., 100000)'
    )
    parser.add_argument(
        '--list-datasets',
        action='store_true',
        help='List available datasets and exit'
    )
    
    args = parser.parse_args()
    
    # List datasets and exit
    if args.list_datasets:
        list_available_datasets()
        return
    
    # Load dataset
    print(f"\n{'='*80}")
    print(f"XGBoost vs LightGBM Comparison")
    print(f"Dataset: {args.dataset}")
    print(f"{'='*80}\n")
    
    if args.dataset == 'higgs' and args.higgs_sample:
        X_train, X_test, y_train, y_test = load_dataset(
            args.dataset,
            sample_size=args.higgs_sample
        )
    else:
        X_train, X_test, y_train, y_test = load_dataset(args.dataset)
    
    # Train and evaluate XGBoost
    print("\n" + "-"*80)
    print("Training XGBoost Baseline...")
    print("-"*80)
    xgb = get_xgboost_model()
    xgb.fit(X_train, y_train)
    xgb_results = evaluate_model(xgb, X_test, y_test)
    print_metrics("XGBoost Baseline", xgb_results)

    # Train and evaluate LightGBM
    print("\n" + "-"*80)
    print("Training LightGBM Baseline...")
    print("-"*80)
    lgbm = get_lightgbm_model()
    lgbm.fit(X_train, y_train)
    lgbm_results = evaluate_model(lgbm, X_test, y_test)
    print_metrics("LightGBM Baseline", lgbm_results)

    # Run ablation study
    print("\n" + "="*80)
    print("Running Ablation Study...")
    print("="*80)
    df_results = run_ablation(X_train, X_test, y_train, y_test, dataset_name=args.dataset)
    
    # Print summary comparison
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80 + "\n")
    summary = compare_models_summary(df_results)
    print(summary)
    
    print(f"\n{'='*80}")
    print("✅ Analysis Complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="XGBoost vs LightGBM Comparative Study"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Skip the ablation study (faster execution)"
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        help="Apply StandardScaler to features"
    )
    
    args = parser.parse_args()
    main(seed=args.seed, skip_ablation=args.skip_ablation, scale=args.scale)
