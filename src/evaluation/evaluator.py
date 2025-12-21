import time
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, 
    recall_score, roc_auc_score
)
from typing import Dict, Optional


def evaluate_model(
    model, 
    X_test, 
    y_test, 
    *, 
    include_timing: bool = False,
    return_proba: bool = False,
    verbose: bool = False
) -> Dict[str, float]:
    """Evaluate a fitted model on a test set.

    Args:
        model: Fitted estimator.
        X_test: Test features.
        y_test: Test labels.
        include_timing: If True, include predict_time_seconds.
        return_proba: If True, also return probability predictions.
        verbose: If True, print warnings.

    Returns:
        Dict of metrics. Optionally includes 'y_proba' if return_proba=True.
    """
    t0 = time.perf_counter()
    preds = model.predict(X_test)
    predict_time = time.perf_counter() - t0

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds, zero_division=0),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
    }

    score = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)
            if proba.shape[1] >= 2:
                score = proba[:, 1]
                if return_proba:
                    metrics["y_proba"] = proba
        except (AttributeError, IndexError, ValueError):
            score = None
    if score is None and hasattr(model, "decision_function"):
        try:
            score = model.decision_function(X_test)
            if len(score.shape) > 1 and score.shape[1] > 1:
                score = score[:, 1] if score.shape[1] >= 2 else score[:, 0]
        except (AttributeError, ValueError):
            score = None

    if score is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_test, score)
        except ValueError:
            if verbose:
                print(f"Warning: Could not compute ROC-AUC")
            metrics["roc_auc"] = np.nan

    if include_timing:
        metrics["predict_time_seconds"] = predict_time

    return metrics