from lightgbm import LGBMClassifier
from typing import Optional, Dict, Any


def get_lightgbm_model(
    params: Optional[Dict[str, Any]] = None, 
    *, 
    random_state: int = 42, 
    n_jobs: int = -1,
    verbosity: int = -1,
    handle_imbalance: bool = False
) -> LGBMClassifier:
    """
    Factory for LightGBM classifier with optional imbalance handling.
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

    if handle_imbalance:
        default_params["is_unbalance"] = True
        # alternative: default_params["class_weight"] = "balanced"

    if params:
        default_params.update(params)

    return LGBMClassifier(**default_params)