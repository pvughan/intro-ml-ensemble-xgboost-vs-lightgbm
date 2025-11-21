from lightgbm import LGBMClassifier

def get_lightgbm_model(params=None):
    if params is None:
        params = {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "subsample": 0.9
        }
    return LGBMClassifier(**params)
