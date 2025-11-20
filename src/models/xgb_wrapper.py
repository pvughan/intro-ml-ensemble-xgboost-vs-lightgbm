import xgboost as xgb

def build_xgb(params=None):
    if params is None:
        params = {"n_estimators": 200, "max_depth": 6}
    return xgb.XGBClassifier(**params)
