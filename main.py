import sys
import argparse
from src.data.loader import load_data
from src.models.xgboost_model import get_xgboost_model
from src.models.lightgbm_model import get_lightgbm_model
from src.evaluation.evaluator import evaluate_model
from src.evaluation.ablation import run_ablation
from src.utils.metrics import print_metrics


def main(seed: int = 42, skip_ablation: bool = False, scale: bool = False):
    """Main execution function.

    Args:
        seed: Random seed for reproducibility.
        skip_ablation: If True, skip the ablation study.
        scale: If True, apply feature scaling.
    """
    try:
        print("=" * 60)
        print("XGBoost vs LightGBM Comparative Study")
        print("=" * 60)
        
        # Load data
        print("\n[1/3] Loading data...")
        X_train, X_test, y_train, y_test = load_data(
            random_state=seed, 
            scale=scale
        )
        print(f"  Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"  Test set: {X_test.shape[0]} samples")

        # Train and evaluate XGBoost
        print("\n[2/3] Training and evaluating models...")
        print("\n  Training XGBoost baseline...")
        xgb = get_xgboost_model(random_state=seed, verbosity=0)
        xgb.fit(X_train, y_train)
        xgb_results = evaluate_model(xgb, X_test, y_test)
        print_metrics("XGBoost Baseline", xgb_results)

        # Train and evaluate LightGBM
        print("\n  Training LightGBM baseline...")
        lgbm = get_lightgbm_model(random_state=seed, verbosity=-1)
        lgbm.fit(X_train, y_train)
        lgbm_results = evaluate_model(lgbm, X_test, y_test)
        print_metrics("LightGBM Baseline", lgbm_results)

        # Ablation study
        if not skip_ablation:
            print("\n[3/3] Running ablation study...")
            df = run_ablation(
                X_train, X_test, y_train, y_test, 
                random_state=seed,
                verbose=True
            )
            print(f"\nSaved ablation table with {len(df)} rows.")
        else:
            print("\n[3/3] Skipping ablation study (--skip-ablation flag set)")

        print("\n" + "=" * 60)
        print("Execution complete!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nWarning: Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


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
