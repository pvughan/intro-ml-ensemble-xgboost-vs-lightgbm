"""
Run ablation experiments: grid search or randomized search over pre-defined parameter grids.
Example:
python src/ablation/ablation_runner.py --dataset wine --algo xgb --mode grid --out_dir results/ablation_wine
"""
import argparse
import os
import yaml
from data.data_loader import load_dataset
from models.xgb_model import XGBModel
from models.lgb_model import LGBModel
from utils.seed import set_seed
from utils.logger import get_logger
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
import joblib
import json
import numpy as np

logger = get_logger("ablation")

def load_config(path="config/default.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wine")
    parser.add_argument("--algo", type=str, choices=["xgb","lgb","both"], default="both")
    parser.add_argument("--mode", type=str, choices=["grid","random"], default="grid")
    parser.add_argument("--n_iter", type=int, default=30, help="for randomized search")
    parser.add_argument("--out_dir", type=str, default="results/ablation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--target_col", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    X, y, *_ = load_dataset(args.dataset, csv_path=args.csv_path, target_col=args.target_col)
    cv_cfg = cfg.get("cv", {})
    skf = StratifiedKFold(n_splits=cv_cfg.get("n_splits",5), shuffle=cv_cfg.get("shuffle",True), random_state=cv_cfg.get("random_state",42))

    results = {}
    if args.algo in ("xgb","both"):
        xgb_cfg = cfg.get("xgb", {})
        base_params = xgb_cfg.get("params", {})
        grid = xgb_cfg.get("default_grid", {})
        from xgboost import XGBClassifier
        est = XGBClassifier(**base_params)
        if args.mode == "grid":
            search = GridSearchCV(est, grid, cv=skf, scoring="accuracy", n_jobs=-1, verbose=1)
        else:
            from scipy.stats import randint
            # simple randomized search using randint; user may customize
            search = RandomizedSearchCV(est, grid, n_iter=args.n_iter, cv=skf, scoring="accuracy", random_state=args.seed, n_jobs=-1, verbose=1)
        logger.info("Running XGBoost search...")
        search.fit(X, y)
        results['xgb'] = {
            'best_params': search.best_params_,
            'best_score': float(search.best_score_)
        }
        joblib.dump(search.best_estimator_, os.path.join(args.out_dir, "best_xgb.joblib"))

    if args.algo in ("lgb","both"):
        lgb_cfg = cfg.get("lgb", {})
        base_params = lgb_cfg.get("params", {})
        grid = lgb_cfg.get("default_grid", {})
        from lightgbm import LGBMClassifier
        est = LGBMClassifier(**base_params)
        if args.mode == "grid":
            search = GridSearchCV(est, grid, cv=skf, scoring="accuracy", n_jobs=-1, verbose=1)
        else:
            from scipy.stats import randint
            search = RandomizedSearchCV(est, grid, n_iter=args.n_iter, cv=skf, scoring="accuracy", random_state=args.seed, n_jobs=-1, verbose=1)
        logger.info("Running LightGBM search...")
        search.fit(X, y)
        results['lgb'] = {
            'best_params': search.best_params_,
            'best_score': float(search.best_score_)
        }
        joblib.dump(search.best_estimator_, os.path.join(args.out_dir, "best_lgb.joblib"))

    with open(os.path.join(args.out_dir, "ablation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Ablation finished. Results saved to %s", args.out_dir)

if __name__ == "__main__":
    main()
