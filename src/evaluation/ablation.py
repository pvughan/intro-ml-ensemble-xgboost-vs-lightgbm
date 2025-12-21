import os
import time
import itertools
import pandas as pd
from typing import Optional

from src.models.xgboost_model import get_xgboost_model
from src.models.lightgbm_model import get_lightgbm_model
from src.evaluation.evaluator import evaluate_model


def run_ablation(
    X_train, X_test, y_train, y_test, 
    *, 
    random_state: int = 42, 
    output_csv: Optional[str] = None,
    verbose: bool = True
):
    """Run hyperparameter ablation study over both models.

    Returns:
        pandas.DataFrame with one row per (model, config).
    """

    shared_search_space = {
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [100, 300, 600],
        "subsample": [0.8, 1.0],
    }
    xgb_specific = {
        "max_depth": [3, 6],
        "colsample_bytree": [0.8, 1.0],
    }
    lgbm_specific = {
        "num_leaves": [15, 31, 63],
        "colsample_bytree": [0.8, 1.0],
    }

    def _iter_grid(d: dict):
        keys = list(d.keys())
        for values in itertools.product(*[d[k] for k in keys]):
            yield dict(zip(keys, values))

    def _count_grid(d: dict) -> int:
        count = 1
        for values in d.values():
            count *= len(values)
        return count

    # Calculate total number of experiments for progress tracking
    num_shared_configs = _count_grid(shared_search_space)
    num_xgb_configs = _count_grid(xgb_specific)
    num_lgbm_configs = _count_grid(lgbm_specific)
    total_experiments = num_shared_configs * (num_xgb_configs + num_lgbm_configs)
    
    if verbose:
        print(f"Running ablation study: {total_experiments} total experiments")
        print(f"  - Shared configs: {num_shared_configs}")
        print(f"  - XGBoost configs: {num_xgb_configs}")
        print(f"  - LightGBM configs: {num_lgbm_configs}")

    records: list[dict] = []
    experiment_count = 0

    for shared_params in _iter_grid(shared_search_space):
        for model_name in ("XGBoost", "LightGBM"):
            if model_name == "XGBoost":
                specific_space = xgb_specific
                builder = get_xgboost_model
            else:
                specific_space = lgbm_specific
                builder = get_lightgbm_model

            for specific_params in _iter_grid(specific_space):
                params = {**shared_params, **specific_params}
                experiment_count += 1

                if verbose and experiment_count % 10 == 0:
                    progress = (experiment_count / total_experiments) * 100
                    print(f"  Progress: {experiment_count}/{total_experiments} ({progress:.1f}%)")

                try:
                    t0 = time.perf_counter()
                    model = builder(params, random_state=random_state)
                    model.fit(X_train, y_train)
                    fit_time = time.perf_counter() - t0

                    metrics = evaluate_model(model, X_test, y_test, include_timing=True)

                    records.append({
                        "model": model_name,
                        **params,
                        "fit_time_seconds": fit_time,
                        **metrics,
                    })
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Error in experiment {experiment_count} ({model_name}): {str(e)}")
                    continue

    if not records:
        raise ValueError("No successful experiments completed. Check error messages above.")
    
    df = pd.DataFrame(records)

    if output_csv is None:
        src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        output_csv = os.path.join(src_dir, "ablation_results.csv")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    if verbose:
        print(f"\nAblation study complete: {len(df)} successful experiments")
        print(f"   Results saved to: {output_csv}")

    return df