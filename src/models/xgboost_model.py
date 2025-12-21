from xgboost import XGBClassifier
from typing import Optional, Dict, Any


def get_xgboost_model(
    params: Optional[Dict[str, Any]] = None, 
    *, 
    random_state: int = 42, 
    n_jobs: int = -1,
    verbosity: int = 0
) -> XGBClassifier:
    """Factory for an XGBoost classifier.

    Args:
        params: Optional hyperparameter dict to override defaults.
        random_state: Seed for reproducibility.
        n_jobs: Parallelism control (-1 for all cores).
        verbosity: Verbosity level (0=silent, 1=warning, 2=info, 3=debug).

    Returns:
        Configured XGBClassifier instance.
    """
    default_params = {
        "n_estimators": 200,
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 1.0,
        "random_state": random_state,
        "n_jobs": n_jobs,
        "verbosity": verbosity,
        "eval_metric": "logloss",
        "use_label_encoder": False,
    }

    if params:
        default_params.update(params)

    return XGBClassifier(**default_params)
