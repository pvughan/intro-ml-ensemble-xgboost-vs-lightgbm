from xgboost import XGBClassifier

def get_xgboost_model(params=None):
    if params is None:
        params = {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 6,
            "subsample": 0.9
        }
    return XGBClassifier(**params)
