from data.loader import load_data
from models.xgboost_model import get_xgboost_model
from models.lightgbm_model import get_lightgbm_model
from evaluation.evaluator import evaluate_model
from evaluation.ablation import run_ablation
from utils.metrics import print_metrics

def main():
    X_train, X_test, y_train, y_test = load_data()

    xgb = get_xgboost_model()
    xgb.fit(X_train, y_train)
    xgb_results = evaluate_model(xgb, X_test, y_test)
    print_metrics("XGBoost Baseline", xgb_results)

    lgbm = get_lightgbm_model()
    lgbm.fit(X_train, y_train)
    lgbm_results = evaluate_model(lgbm, X_test, y_test)
    print_metrics("LightGBM Baseline", lgbm_results)

    print("\n=== Running Ablation Study ===")
    ab_results = run_ablation(X_train, X_test, y_train, y_test)
    for model_name, params, metrics in ab_results:
        print(f"\nModel: {model_name}, Params: {params}")
        print_metrics("Metrics", metrics)

if __name__ == "__main__":
    main()
