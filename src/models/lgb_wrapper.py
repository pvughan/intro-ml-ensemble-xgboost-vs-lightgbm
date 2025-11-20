import lightgbm as lgb

def build_lgb(params=None):
    if params is None:
        params = {"num_leaves": 31, "n_estimators": 200}
    return lgb.LGBMClassifier(**params)
