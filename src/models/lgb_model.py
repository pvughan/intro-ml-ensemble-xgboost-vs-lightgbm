import lightgbm as lgb
import joblib

class LGBModel:
    def __init__(self, task="classification", **params):
        self.task = task
        if task == "classification":
            self.model = lgb.LGBMClassifier(**params)
        else:
            self.model = lgb.LGBMRegressor(**params)

    def fit(self, X, y, **fit_params):
        self.model.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            return None

    def get_sklearn_estimator(self):
        return self.model

    def save(self, path):
        joblib.dump(self.model, path)

    @staticmethod
    def load(path):
        m = joblib.load(path)
        inst = LGBModel()
        inst.model = m
        return inst
