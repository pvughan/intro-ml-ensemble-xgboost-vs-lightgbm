"""
Load a saved model and run evaluation on chosen dataset.
"""
import argparse
import joblib
from data.data_loader import load_dataset
from evaluation.metrics import compute_classification_metrics
from utils.seed import set_seed
from utils.logger import get_logger
import numpy as np

logger = get_logger("evaluate")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="breast_cancer")
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--target_col", type=str, default=None)
    args = parser.parse_args()

    set_seed(42)
    X, y, _, _ = load_dataset(args.dataset, csv_path=args.csv_path, target_col=args.target_col)

    model = joblib.load(args.model_path)
    try:
        y_pred = model.predict(X)
    except Exception as e:
        logger.error("Model prediction failed: %s", e)
        raise

    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X)
        except Exception:
            y_prob = None

    metrics = compute_classification_metrics(y, y_pred, y_prob)
    print("Evaluation results:", metrics)

if __name__ == "__main__":
    main()
