from models.xgboost_model import get_xgboost_model
from models.lightgbm_model import get_lightgbm_model
from evaluation.evaluator import evaluate_model

def run_ablation(X_train, X_test, y_train, y_test):

    search_space = {
        "learning_rate": [0.01, 0.1],
        "n_estimators": [100, 300]
    }

    results = []

    for lr in search_space["learning_rate"]:
        for n in search_space["n_estimators"]:
            
            xgb_params = {"learning_rate": lr, "n_estimators": n}
            xgb = get_xgboost_model(xgb_params)
            xgb.fit(X_train, y_train)
            xgb_metrics = evaluate_model(xgb, X_test, y_test)
            results.append(("XGBoost", xgb_params, xgb_metrics))

            lgbm_params = {"learning_rate": lr, "n_estimators": n}
            lgbm = get_lightgbm_model(lgbm_params)
            lgbm.fit(X_train, y_train)
            lgbm_metrics = evaluate_model(lgbm, X_test, y_test)
            results.append(("LightGBM", lgbm_params, lgbm_metrics))

    return results
