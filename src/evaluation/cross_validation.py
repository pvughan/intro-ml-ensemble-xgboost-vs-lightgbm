import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
import pandas as pd


def cross_validate_model(model_builder, X, y, k=5, model_name="UnknownModel"):
    """
    Performs Stratified K-Fold Cross Validation
    Returns mean ± std metrics
    """

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    records = []

    fold = 0
    for train_idx, test_idx in skf.split(X, y):
        fold += 1
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = model_builder()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_test)[:, 1]
            except:
                proba = None

        fold_metrics = {
            "fold": fold,
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
        }

        if proba is not None:
            fold_metrics["roc_auc"] = roc_auc_score(y_test, proba)
        else:
            fold_metrics["roc_auc"] = np.nan

        records.append(fold_metrics)

    df = pd.DataFrame(records)

    summary = df.mean().round(4)
    summary_std = df.std().round(4)

    print(f"\n===== {model_name} CROSS-VALIDATION ({k}-Fold) =====")
    print(df)
    print("\nMean Performance:")
    print(summary)
    print("\nStd Deviation:")
    print(summary_std)

    return df, summary, summary_std