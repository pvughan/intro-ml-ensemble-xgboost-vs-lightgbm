"""
Main training script.
Usage examples:
python src/training/train.py --dataset breast_cancer --model xgb --out_dir results/xgb_breast
python src/training/train.py --dataset wine --model lgb --out_dir results/lgb_wine --do_cv
"""
import argparse
import os
import yaml
from data.data_loader import load_dataset
from models.xgb_model import XGBModel
from models.lgb_model import LGBModel
from models.model_utils import evaluate_cv
from utils.seed import set_seed
from utils.logger import get_logger
import joblib
from sklearn.model_selection import StratifiedKFold
import numpy as np

logger = get_logger("train")

def load_config(path="config/default.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="breast_cancer")
    parser.add_argument("--model", type=str, choices=["xgb","lgb"], default="xgb")
    parser.add_argument("--out_dir", type=str, default="results/run")
    parser.add_argument("--do_cv", action="store_true", help="Run cross-validation before fit")
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--target_col", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    X, y, feature_names, target_names = load_dataset(args.dataset, csv_path=args.csv_path, target_col=args.target_col)
    task = "classification" if len(np.unique(y)) <= 20 else "regression"  # heuristic

    if args.model == "xgb":
        params = cfg.get("xgb", {}).get("params", {})
        model = XGBModel(task=task, **params)
    else:
        params = cfg.get("lgb", {}).get("params", {})
        model = LGBModel(task=task, **params)

    if args.do_cv:
        cv_cfg = cfg.get("cv", {})
        skf = StratifiedKFold(n_splits=cv_cfg.get("n_splits",5), shuffle=cv_cfg.get("shuffle",True), random_state=cv_cfg.get("random_state",42))
        scoring = ['accuracy', 'f1_macro']
        est = model.get_sklearn_estimator()
        logger.info("Running CV...")
        res = evaluate_cv(est, X, y, cv=skf, scoring={'accuracy':'accuracy','f1_macro':'f1_macro'})
        logger.info("CV results: %s", res)

    logger.info("Fitting final model on full training data...")
    model.fit(X, y)
    model_path = os.path.join(args.out_dir, "model.joblib")
    model.save(model_path)
    logger.info("Saved model to %s", model_path)

    # Save basic metrics on training set
    try:
        y_pred = model.predict(X)
        y_prob = None
        if hasattr(model.get_sklearn_estimator(), "predict_proba"):
            y_prob = model.get_sklearn_estimator().predict_proba(X)
        from evaluation.metrics import compute_classification_metrics
        metrics = compute_classification_metrics(y, y_pred, y_prob)
        import json
        with open(os.path.join(args.out_dir, "train_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved training metrics")
    except Exception as e:
        logger.warning("Failed to compute train metrics: %s", e)

if __name__ == "__main__":
    main()
