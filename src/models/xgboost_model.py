from xgboost import XGBClassifier
from typing import Optional, Dict, Any


def get_xgboost_model(
    params: Optional[Dict[str, Any]] = None, 
    *, 
    random_state: int = 42, 
    n_jobs: int = -1,
    verbosity: int = 0,
    scale_pos_weight: Optional[float] = None
) -> XGBClassifier:
    """
    Factory for XGBoost classifier with optional class imbalance control.
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

    if scale_pos_weight is not None:
        default_params["scale_pos_weight"] = scale_pos_weight

    if params:
        default_params.update(params)

    return XGBClassifier(**default_params)