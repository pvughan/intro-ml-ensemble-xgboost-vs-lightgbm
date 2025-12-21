from lightgbm import LGBMClassifier
from typing import Optional, Dict, Any


def get_lightgbm_model(
    params: Optional[Dict[str, Any]] = None, 
    *, 
    random_state: int = 42, 
    n_jobs: int = -1,
    verbosity: int = -1
) -> LGBMClassifier:
    """Factory for a LightGBM classifier.

    Args:
        params: Optional hyperparameter dict to override defaults.
        random_state: Seed for reproducibility.
        n_jobs: Parallelism control (-1 for all cores).
        verbosity: Verbosity level (-1=warning, 0=info, 1=debug).

    Returns:
        Configured LGBMClassifier instance.
    """
    default_params = {
        "n_estimators": 200,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "subsample": 0.9,
        "colsample_bytree": 1.0,
        "random_state": random_state,
        "n_jobs": n_jobs,
        "verbosity": verbosity,
    }

    if params:
        default_params.update(params)

    return LGBMClassifier(**default_params)
