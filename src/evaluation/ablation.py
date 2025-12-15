from models.xgboost_model import get_xgboost_model
from models.lightgbm_model import get_lightgbm_model
from evaluation.evaluator import evaluate_model
import pandas as pd
def run_ablation(X_train, X_test, y_train, y_test):
    search_space = {
        "learning_rate": [0.01, 0.1],
        "n_estimators": [100, 300]
    }

    records = []

    for lr in search_space["learning_rate"]:
        for n in search_space["n_estimators"]:

            for model_name, builder in {
                "XGBoost": get_xgboost_model,
                "LightGBM": get_lightgbm_model
            }.items():

                params = {"learning_rate": lr, "n_estimators": n}
                model = builder(params)
                model.fit(X_train, y_train)

                metrics = evaluate_model(model, X_test, y_test)

                records.append({
                    "model": model_name,
                    "learning_rate": lr,
                    "n_estimators": n,
                    **metrics
                })

    df = pd.DataFrame(records)
    df.to_csv("ablation_results.csv", index=False)

    return df