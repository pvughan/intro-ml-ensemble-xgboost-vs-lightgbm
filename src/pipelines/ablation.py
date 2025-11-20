from src.models.xgb_wrapper import build_xgb
from src.models.lgb_wrapper import build_lgb
from src.data.loader import load_dataset
from sklearn.metrics import accuracy_score

def run_ablation():
    X_train, X_test, y_train, y_test = load_dataset()

    settings = [
        {"n_estimators": 50},
        {"n_estimators": 200},
        {"n_estimators": 500},
    ]

    results = []
    for cfg in settings:
        model = build_xgb(cfg)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        results.append((cfg, acc))

    return results
