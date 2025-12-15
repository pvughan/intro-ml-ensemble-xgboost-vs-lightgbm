from models.xgboost_model import get_xgboost_model
from models.lightgbm_model import get_lightgbm_model
from evaluation.evaluator import evaluate_model
from data.loader import load_data

def main():
    X_train, X_test, y_train, y_test = load_data()

    models = {
        "XGBoost": get_xgboost_model(),
        "LightGBM": get_lightgbm_model()
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate_model(model, X_test, y_test)

    for model, metrics in results.items():
        print(f"\n{model}")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()