from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
import numpy as np

def evaluate_cv(estimator, X, y, cv, scoring=None):
    # scoring can be a dict or list of scorers compatible with sklearn
    res = cross_validate(estimator, X, y, cv=cv, scoring=scoring, return_train_score=False)
    # aggregate to mean/std
    agg = {k: (float(np.mean(v)), float(np.std(v))) for k, v in res.items()}
    return agg
