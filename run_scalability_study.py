"""
Main entry point for scalability study
Runs comprehensive multi-dataset comparison of XGBoost vs LightGBM
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from evaluation.scalability_analysis import run_multi_dataset_analysis
from data.loader import list_available_datasets


def main():
    parser = argparse.ArgumentParser(
        description='XGBoost vs LightGBM Scalability Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on all datasets (WARNING: HIGGS is 11M samples!)
  python run_scalability_study.py --all
  
  # Run on specific datasets
  python run_scalability_study.py --datasets breast_cancer creditcard_fraud
  
  # Run with HIGGS subset (recommended)
  python run_scalability_study.py --all --higgs-sample 100000
  
  # List available datasets
  python run_scalability_study.py --list
        """
    )
    
    parser.add_argument(
        '--datasets', 
        nargs='+', 
        help='Specific datasets to analyze (e.g., breast_cancer creditcard_fraud higgs)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run analysis on all available datasets'
    )
    
    parser.add_argument(
        '--higgs-sample',
        type=int,
        default=None,
        help='Sample size for HIGGS dataset (e.g., 100000 for 100K samples). If not specified, uses full 11M samples.'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available datasets and exit'
    )
    
    args = parser.parse_args()
    
    # List datasets and exit
    if args.list:
        list_available_datasets()
        return
    
    # Determine which datasets to run
    if args.all:
        dataset_names = None  # Will use all datasets
        print("📊 Running analysis on ALL datasets")
        if args.higgs_sample:
            print(f"   HIGGS will use {args.higgs_sample:,} samples (subset)")
    elif args.datasets:
        dataset_names = args.datasets
        print(f"📊 Running analysis on: {', '.join(dataset_names)}")
    else:
        # Default: run on small and medium datasets
        dataset_names = ['breast_cancer', 'creditcard_fraud']
        print("📊 Running analysis on default datasets: breast_cancer, creditcard_fraud")
        print("   (Use --all to include HIGGS dataset)")
    
    print("\n" + "="*80)
    print("STARTING SCALABILITY STUDY")
    print("="*80 + "\n")
    
    # Warning for large datasets
    if dataset_names is None or 'higgs' in dataset_names:
        if args.higgs_sample is None:
            print("⚠️  WARNING: You are about to process the full HIGGS dataset (11M samples)")
            print("   This will take significant time and memory (16GB+ RAM recommended)")
            response = input("\nContinue with full HIGGS dataset? (yes/no): ").strip().lower()
            if response != 'yes':
                print("\n❌ Cancelled. Use --higgs-sample to specify a subset size.")
                return
    
    # Run analysis
    try:
        results = run_multi_dataset_analysis(
            dataset_names=dataset_names,
            higgs_sample_size=args.higgs_sample
        )
        
        print("\n" + "="*80)
        print("✅ SCALABILITY STUDY COMPLETE")
        print("="*80)
        print(f"\nResults saved to:")
        print(f"  - Individual dataset results: ablation_results_<dataset>.csv")
        print(f"  - Consolidated results: scalability_results.csv")
        print(f"\nTo generate visualizations, run:")
        print(f"  python src/evaluation/visualize.py")
        print()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
